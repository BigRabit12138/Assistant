import time
import logging

from typing import Any

import tiktoken

from assistant.memory.graphrag_v1.query.context_builder.builders import LocalContextBuilder
from assistant.memory.graphrag_v1.query.context_builder.conversation_history import (
    ConversationHistory,
)
from assistant.memory.graphrag_v1.query.llm.base import BaseLLM, BaseLLMCallback
from assistant.memory.graphrag_v1.query.llm.text_utils import num_tokens
from assistant.memory.graphrag_v1.query.question_gen.base import BaseQuestionGen, QuestionResult
from assistant.memory.graphrag_v1.query.question_gen.system_prompt import QUESTION_SYSTEM_PROMPT

log = logging.getLogger(__name__)


class LocalQuestionGen(BaseQuestionGen):
    def __init__(
            self,
            llm: BaseLLM,
            context_builder: LocalContextBuilder,
            token_encoder: tiktoken.Encoding | None = None,
            system_prompt: str = QUESTION_SYSTEM_PROMPT,
            callbacks: list[BaseLLMCallback] | None = None,
            llm_params: dict[str, Any] | None = None,
            context_builder_params: dict[str, Any] | None = None,
    ):
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=context_builder_params,
        )
        self.system_prompt = system_prompt
        self.callbacks = callbacks

    async def agenerate(
            self,
            question_history: list[str],
            context_data: str | None,
            question_count: int,
            **kwargs
    ) -> QuestionResult:
        start_time = time.time()

        if len(question_history) == 0:
            question_text = ""
            conversation_history = None
        else:
            question_text = question_history[-1]
            history = [
                {"role": "user", "content": query} for query in question_history[:-1]
            ]
            conversation_history = ConversationHistory.from_list(history)
        if context_data is None:
            context_data, context_records = self.context_builder.build_context(
                query=question_text,
                conversation_history=conversation_history,
                **kwargs,
                **self.context_builder_params,
            )
        else:
            context_records = {"context_data": context_data}
        log.info(f"GENERATE QUESTION: {start_time}. LAST QUESTION: {question_text}")
        system_prompt = ""
        try:
            system_prompt = self.system_prompt.format(
                context_data=context_data,
                question_count=question_count,
            )
            question_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ]

            response = await self.llm.agenerate(
                messages=question_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return QuestionResult(
                response=response.split("\n"),
                context_data={
                    "question_context": question_text,
                    **context_records,
                },
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(system_prompt, self.token_encoder),
            )
        except Exception:
            log.exception("Exception in generating question")
            return QuestionResult(
                response=[],
                context_data=context_records,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(system_prompt, self.token_encoder),
            )

    def generate(
            self,
            question_history: list[str],
            context_data: str | None,
            question_count: int,
            **kwargs,
    ) -> QuestionResult:
        start_time = time.time()
        if len(question_history) == 0:
            question_text = ""
            conversation_history = None
        else:
            question_text = question_history[-1]
            history = [
                {"role": "user", "context": query} for query in question_history[:-1]
            ]
            conversation_history = ConversationHistory.from_list(history)

        if context_data is None:
            context_data, context_records = self.context_builder.build_context(
                query=question_text,
                conversation_history=conversation_history,
                **kwargs,
                **self.context_builder_params,
            )
        else:
            context_records = {"context_data": context_data}
        log.info(
            f"GENERATE QUESTION: {start_time}. QUESTION HISTORY: {question_text}"
        )
        system_prompt = ""
        try:
            system_prompt = self.system_prompt.format(
                context_data=context_data,
                question_count=question_count,
            )
            question_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ]
            response = self.llm.generate(
                messages=question_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return QuestionResult(
                response=response.split("\n"),
                context_data={
                    "question_context": question_text,
                    **context_records,
                },
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(system_prompt, self.token_encoder),
            )
        except Exception:
            log.exception("Exception in generating questions")
            return QuestionResult(
                response=[],
                context_data=context_records,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(system_prompt, self.token_encoder),
            )
