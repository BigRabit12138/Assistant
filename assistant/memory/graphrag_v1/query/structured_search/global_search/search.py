import json
import time
import asyncio
import logging

from typing import Any
from dataclasses import dataclass
from collections.abc import AsyncGenerator

import tiktoken
import pandas as pd

from assistant.memory.graphrag_v1.llm.openai.utils import try_parse_json_object
from assistant.memory.graphrag_v1.query.context_builder.builders import GlobalContextBuilder
from assistant.memory.graphrag_v1.query.context_builder.conversation_history import (
    ConversationHistory,
)
from assistant.memory.graphrag_v1.query.llm.base import BaseLLM
from assistant.memory.graphrag_v1.query.llm.text_utils import num_tokens
from assistant.memory.graphrag_v1.query.structured_search.base import BaseSearch, SearchResult
from assistant.memory.graphrag_v1.query.structured_search.global_search.callbacks import (
    GlobalSearchLLMCallback,
)
from assistant.memory.graphrag_v1.query.structured_search.global_search.map_system_prompt import (
    MAP_SYSTEM_PROMPT,
)
from assistant.memory.graphrag_v1.query.structured_search.global_search.reduce_system_prompt import (
    NO_DATA_ANSWER,
    REDUCE_SYSTEM_PROMPT,
    GENERAL_KNOWLEDGE_INSTRUCTION,
)

DEFAULT_MAP_LLM_PARAMS = {
    "max_tokens": 1000,
    "temperature": 0.0,
}

DEFAULT_REDUCE_LLM_PARAMS = {
    "max_tokens": 2000,
    "temperature": 0.0,
}

log = logging.getLogger(__name__)


@dataclass
class GlobalSearchResult(SearchResult):
    map_responses: list[SearchResult]
    reduce_context_data: str | list[pd.DataFrame] | dict[str, pd.DataFrame]
    reduce_context_text: str | list[str] | dict[str, str]


class GlobalSearch(BaseSearch):
    def __init__(
            self,
            llm: BaseLLM,
            context_builder: GlobalContextBuilder,
            token_encoder: tiktoken.Encoding | None = None,
            map_system_prompt: str = MAP_SYSTEM_PROMPT,
            reduce_system_prompt: str = REDUCE_SYSTEM_PROMPT,
            response_type: str = "multiple paragraphs",
            allow_general_knowledge: bool = False,
            general_knowledge_inclusion_prompt: str = GENERAL_KNOWLEDGE_INSTRUCTION,
            json_mode: bool = True,
            callbacks: list[GlobalSearchLLMCallback] | None = None,
            max_data_tokens: int = 8000,
            map_llm_params: dict[str, Any] = DEFAULT_MAP_LLM_PARAMS,
            reduce_llm_params: dict[str, Any] = DEFAULT_REDUCE_LLM_PARAMS,
            context_builder_params: dict[str, Any] | None = None,
            concurrent_coroutines: int = 32,
    ):
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            context_builder_params=context_builder_params,
        )
        self.map_system_prompt = map_system_prompt
        self.reduce_system_prompt = reduce_system_prompt
        self.response_type = response_type
        self.allow_general_knowledge = allow_general_knowledge
        self.general_knowledge_inclusion_prompt = general_knowledge_inclusion_prompt
        self.callbacks = callbacks
        self.max_data_tokens = max_data_tokens

        self.map_llm_params = map_llm_params
        self.reduce_llm_params = reduce_llm_params
        if json_mode:
            self.map_llm_params["response_format"] = {"type": "json_object"}
        else:
            self.map_llm_params.pop("response_format", None)

        self.semaphore = asyncio.Semaphore(concurrent_coroutines)

    async def astream_search(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
    ) -> AsyncGenerator:
        context_chunks, contex_records = self.context_builder.build_context(
            conversation_history=conversation_history, **self.context_builder_params,
        )
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_map_response_start(context_chunks)
        map_responses = await asyncio.gather(*[
            self._map_response_single_batch(
                context_data=data,
                query=query,
                **self.map_llm_params,
            )
            for data in context_chunks
        ])
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_map_response_end(map_responses)

        yield contex_records
        async for response in self._stream_reduce_response(
            map_responses=map_responses,
            query=query,
            **self.reduce_llm_params,
        ):
            yield response

    async def asearch(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
            **kwargs: Any,
    ) -> GlobalSearchResult:
        start_time = time.time()
        context_chunks, context_records = self.context_builder.build_context(
            conversation_history=conversation_history,
            **self.context_builder_params,
        )

        if self.callbacks:
            for callback in self.callbacks:
                callback.on_map_response_start(context_chunks)
        map_responses = await asyncio.gather(*[
            self._map_response_single_batch(
                context_data=data, query=query, **self.map_llm_params
            )
            for data in context_chunks
        ])
        if self.callbacks:
            for callback in self.callbacks:
                callback.on_map_response_end(map_responses)
        map_llm_calls = sum(response.llm_calls for response in map_responses)
        map_prompt_tokens = sum(response.prompt_tokens for response in map_responses)

        reduce_response = await self._reduce_response(
            map_responses=map_responses,
            query=query,
            **self.reduce_llm_params,
        )

        return GlobalSearchResult(
            response=reduce_response.response,
            context_data=context_records,
            context_text=context_chunks,
            map_responses=map_responses,
            reduce_context_data=reduce_response.context_data,
            reduce_context_text=reduce_response.context_text,
            completion_time=time.time() - start_time,
            llm_calls=map_llm_calls + reduce_response.llm_calls,
            prompt_tokens=map_prompt_tokens + reduce_response.prompt_tokens,
        )

    def search(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
            **kwargs: Any,
    ) -> GlobalSearchResult:
        return asyncio.run(self.asearch(query, conversation_history))

    async def _map_response_single_batch(
            self,
            context_data: str,
            query: str,
            **llm_kwargs,
    ) -> SearchResult:
        start_time = time.time()
        search_prompt = ""
        try:
            search_prompt = self.map_system_prompt.format(
                context_data=context_data
            )
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]
            async with self.semaphore:
                search_response = await self.llm.agenerate(
                    messages=search_messages,
                    streaming=False,
                    **llm_kwargs,
                )
                log.info(f"Map response: {search_response}")
            try:
                processed_response = self.parse_search_response(search_response)
            except ValueError:
                try:
                    processed_response = self.parse_search_response(search_response)
                except ValueError:
                    log.warning(
                        "Warning: Error parsing search response json - skipping this batch"
                    )
                    processed_response = []
            return SearchResult(
                response=processed_response,
                context_data=context_data,
                context_text=context_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )
        except Exception:
            log.exception("Exception in _map_response_single_batch")
            return SearchResult(
                response=[{"answer": "", "score": 0}],
                context_data=context_data,
                context_text=context_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

    @staticmethod
    def parse_search_response(
            self,
            search_response: str
    ) -> list[dict[str, Any]]:
        search_response, _j = try_parse_json_object(search_response)
        if _j == {}:
            return [{"answer": "", "score": 0}]

        parsed_elements = json.loads(search_response).get("points")
        if not parsed_elements or not isinstance(parsed_elements, list):
            return [{"answer": "", "score": 0}]

        return [
            {
                "answer": element["description"],
                "score": int(element["score"]),
            }
            for element in parsed_elements
            if "description" in element and "score" in element
        ]

    async def _reduce_response(
            self,
            map_responses: list[SearchResult],
            query: str,
            **llm_kwargs,
    ) -> SearchResult:
        text_data = ""
        search_prompt = ""
        start_time = time.time()
        try:
            key_points = []
            for index, response in enumerate(map_responses):
                if not isinstance(response.response, list):
                    continue
                for element in response.response:
                    if not isinstance(element, dict):
                        continue
                    if "answer" not in element or "score" not in element:
                        continue
                    key_points.append({
                        "analyst": index,
                        "answer": element["answer"],
                        "score": element["score"],
                    })

            filtered_key_points = [
                point
                for point in key_points
                if point["score"] > 0
            ]

            if len(filtered_key_points) == 0 and not self.allow_general_knowledge:
                log.warning(
                    "Warning: All map responses have score 0 (i.e., not relevant information found from the dataset), \
returning a canned 'I do not know' answer. You can try enabling `allow_general_knowledge` to encourage the LLM \
to incorporate relevant general knowledge, at the risk of increasing hallucinations."
                )
                return SearchResult(
                    response=NO_DATA_ANSWER,
                    context_data="",
                    context_text="",
                    completion_time=time.time() - start_time,
                    llm_calls=0,
                    prompt_tokens=0,
                )
            filtered_key_points = sorted(
                filtered_key_points,
                key=lambda x: x["score"],
                reverse=True,
            )

            data = []
            total_tokens = 0
            for point in filtered_key_points:
                formatted_response_data = []
                formatted_response_data.append(
                    f'----Analyst {point["analyst"] + 1}----'
                )
                formatted_response_data.append(
                    f'Importance Score: {point["score"]}'
                )
                formatted_response_data.append(point["answer"])
                formatted_response_text = "\n".join(formatted_response_data)
                if (
                    total_tokens
                    + num_tokens(formatted_response_text, self.token_encoder)
                    > self.max_data_tokens
                ):
                    break
                data.append(formatted_response_text)
                total_tokens += num_tokens(formatted_response_text, self.token_encoder)
            text_data = "\n\n".join(data)

            search_prompt = self.reduce_system_prompt.format(
                report_data=text_data,
                response_type=self.response_type,
            )
            if self.allow_general_knowledge:
                search_prompt += "\n" + self.general_knowledge_inclusion_prompt
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            search_response = await self.llm.agenerate(
                search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **llm_kwargs,
            )
            return SearchResult(
                response=search_response,
                context_data=text_data,
                context_text=text_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )
        except Exception:
            log.exception("Exception in _reduce_response")
            return SearchResult(
                response="",
                context_data=text_data,
                context_text=text_data,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

    async def _stream_reduce_response(
            self,
            map_responses: list[SearchResult],
            query: str,
            **llm_kwargs,
    ) -> AsyncGenerator[str, None]:
        key_points = []
        for index, response in enumerate(map_responses):
            if not isinstance(response.response, list):
                continue
            for element in response.response:
                if not isinstance(element, dict):
                    continue
                if "answer" not in element or "score" not in element:
                    continue
                key_points.append({
                    "analyst": index,
                    "answer": element["answer"],
                    "score": element["score"],
                })

        filtered_key_points = [
            point
            for point in key_points
            if point["score"] > 0
        ]

        if len(filtered_key_points) == 0 and not self.allow_general_knowledge:
            log.warning(
                "Warning: All map responses have score 0 (i.e., not relevant information found from the dataset), \
returning a canned 'I do not know' answer. You can try enabling `allow_general_knowledge` to encourage the LLM \
to incorporate relevant general knowledge, at the risk of increasing hallucinations."
            )
            yield NO_DATA_ANSWER
            return

        filtered_key_points = sorted(
            filtered_key_points,
            key=lambda x: x["score"],
            reverse=True,
        )

        data = []
        total_tokens = 0
        for point in filtered_key_points:
            formatted_response_data = [
                f'-----Analyst {point["analyst"] + 1}----',
                f'Importance Score: {point["score"]}',
                point["answer"],
            ]
            formatted_response_text = "\n".join(formatted_response_data)
            if (
                total_tokens + num_tokens(formatted_response_text, self.token_encoder)
                > self.max_data_tokens
            ):
                break
            data.append(formatted_response_text)
            total_tokens += num_tokens(formatted_response_text, self.token_encoder)

        text_data = "\n\n".join(data)
        search_prompt = self.reduce_system_prompt.format(
            report_data=text_data,
            response_type=self.response_type,
        )
        if self.allow_general_knowledge:
            search_prompt += "\n" + self.general_knowledge_inclusion_prompt
        search_messages = [
            {"role": "system", "content": search_prompt},
            {"role": "user", "content": query},
        ]
        async for resp in self.llm.astream_generate(
            search_messages,
            callbacks=self.callbacks,
            **llm_kwargs,
        ):
            yield resp
