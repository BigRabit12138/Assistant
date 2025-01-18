from typing import Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

import tiktoken
import pandas as pd

from assistant.memory.graphrag_v1.query.llm.base import BaseLLM
from assistant.memory.graphrag_v1.query.context_builder.builders import (
    LocalContextBuilder,
    GlobalContextBuilder,
)
from assistant.memory.graphrag_v1.query.context_builder.conversation_history import (
    ConversationHistory,
)


@dataclass
class SearchResult:
    response: str | dict[str, Any] | list[dict[str, Any]]
    context_data: str | list[pd.DataFrame] | dict[str, pd.DataFrame]
    context_text: str | list[str] | dict[str, str]
    completion_time: float
    llm_calls: int
    prompt_tokens: int


class BaseSearch(ABC):
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
    def search(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
            **kwargs,
    ) -> SearchResult:
        pass

    @abstractmethod
    async def asearch(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
            **kwargs,
    ) -> SearchResult:
        pass

    @abstractmethod
    def astream_search(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
    ) -> AsyncGenerator[str, None]:
        pass
