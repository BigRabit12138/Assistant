from typing import Any
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator


class BaseLLMCallback:
    def __init__(self):
        self.response = []

    def on_llm_new_token(self, token: str):
        self.response.append(token)


class BaseLLM(ABC):
    @abstractmethod
    def generate(
            self,
            messages: str | list[Any],
            streaming: bool = True,
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> str:
        pass

    @abstractmethod
    def stream_generate(
            self,
            messages: str | list[Any],
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> Generator[str, None, None]:
        pass

    @abstractmethod
    async def agenerate(
            self,
            messages: str | list[Any],
            streaming: bool = True,
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> str:
        pass

    @abstractmethod
    async def astream_generate(
            self,
            messages: str | list[Any],
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        pass


class BaseTextEmbedding(ABC):
    @abstractmethod
    def embed(
            self,
            text: str,
            **kwargs: Any,
    ) -> list[float]:
        pass

    @abstractmethod
    async def aembed(
            self,
            text: str,
            **kwargs: Any,
    ) -> list[float]:
        pass
