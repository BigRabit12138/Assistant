import time
import logging

from typing import Any
from collections.abc import AsyncGenerator

import tiktoken

from assistant.memory.graphrag_v1.query.context_builder.builders import LocalContextBuilder
from assistant.memory.graphrag_v1.query.context_builder.conversation_history import (
    ConversationHistory,
)
from assistant.memory.graphrag_v1.query.llm.base import BaseLLM, BaseLLMCallback
from assistant.memory.graphrag_v1.query.llm.text_utils import num_tokens
from assistant.memory.graphrag_v1.query.structured_search.base import BaseSearch, SearchResult
from assistant.memory.graphrag_v1.query.structured_search.local_search.system_prompt import (
    LOCAL_SEARCH_SYSTEM_PROMPT,
)

DEFAULT_LLM_PARAMS = {
    "max_tokens": 1500,
    "temperature": 0.0,
}

log = logging.getLogger(__name__)


class LocalSearch(BaseSearch):
    def __init__(
            self,
            llm: BaseLLM,
            context_builder: LocalContextBuilder,
            token_encoder: tiktoken.Encoding | None = None,
            system_prompt: str = LOCAL_SEARCH_SYSTEM_PROMPT,
            response_type: str = "multiple paragraphs",
            callbacks: list[BaseLLMCallback] | None = None,
            llm_params: dict[str, Any] = DEFAULT_LLM_PARAMS,
            context_builder_params: dict | None = None,
    ):
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=context_builder_params or {},
        )
        self.system_prompt = system_prompt
        self.callbacks = callbacks
        self.response_type = response_type

    async def asearch(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
            **kwargs,
    ) -> SearchResult:
        start_time = time.time()
        search_prompt = ""

        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        log.info(f"GENERATE ANSWER: {start_time}. QUERY: {query}")
        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text,
                response_type=self.response_type,
            )
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            response = await self.llm.agenerate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )
        except Exception:
            log.exception("Exception in asearch")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

    async def astream_search(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
    ) -> AsyncGenerator:
        start_time = time.time()

        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **self.context_builder_params,
        )

        log.info(f"GENERATE ANSWER: {start_time}. QUERY: {query}")
        search_prompt = self.system_prompt.format(
            context_data=context_text,
            response_type=self.response_type,
        )
        search_messages = [
            {"role": "system", "content": search_prompt},
            {"role": "user", "content": query},
        ]

        yield context_records
        async for response in self.llm.astream_generate(
            messages=search_messages,
            callbacks=self.callbacks,
            **self.llm_params,
        ):
            yield response

    def search(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
            **kwargs,
    ) -> SearchResult:
        start_time = time.time()
        search_prompt = ""
        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        log.info(f"GENERATE ANSWER: {start_time}. QUERY: {query}")
        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text,
                response_type=self.response_type,
            )
            search_messages = [
                {"role": "system", "context": search_prompt},
                {"role": "user", "content": query},
            ]
            response = self.llm.generate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )
        except Exception:
            log.exception("Exception in search")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )
