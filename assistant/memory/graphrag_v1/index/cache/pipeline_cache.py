from __future__ import annotations

from typing import Any
from abc import ABCMeta, abstractmethod


class PipelineCache(metaclass=ABCMeta):

    @abstractmethod
    async def get(self, key: str) -> Any:
        pass

    @abstractmethod
    async def set(
            self,
            key: str,
            value: Any,
            debug_data: dict | None = None
    ) -> None:
        pass

    @abstractmethod
    async def has(self, key: str) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass

    @abstractmethod
    def child(self, name: str) -> PipelineCache:
        pass
