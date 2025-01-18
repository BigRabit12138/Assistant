from typing import Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import tiktoken

from assistant.memory.graphrag_v1.query.llm.base import BaseLLM
from assistant.memory.graphrag_v1.query.context_builder.builders import (
    LocalContextBuilder,
    GlobalContextBuilder,
)


@dataclass
class QuestionResult:
    response: list[str]
    context_data: str | dict[str, Any]
    completion_time: float
    llm_calls: int
    prompt_tokens: int


class BaseQuestionGen(ABC):
    def __init__(
            self,
            llm: BaseLLM,
            context_builder: GlobalContextBuilder | LocalContextBuilder,
            token_encoder: tiktoken.Encoding | None = None,
            llm_params: dict[str, Any] | None = None,
            context_builder_params: dict[str, Any] | None = None,
    ):
        self.llm = llm
        self.context_builder = context_builder
        self.token_encoder = token_encoder
        self.llm_params = llm_params or {}
        self.context_builder_params = context_builder_params or {}

    @abstractmethod
    def generate(
            self,
            question_history: list[str],
            context_data: str | None,
            question_count: int,
            **kwargs,
    ) -> QuestionResult:
        pass

    @abstractmethod
    async def agenerate(
            self,
            question_history: list[str],
            context_data: str | None,
            question_count: int,
            **kwargs
    ) -> QuestionResult:
        pass
