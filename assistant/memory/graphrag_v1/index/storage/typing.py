import re

from typing import Any
from collections.abc import Iterator
from abc import ABCMeta, abstractmethod

from assistant.memory.graphrag_v1.index.progress import ProgressReporter


class PipelineStorage(metaclass=ABCMeta):
    @abstractmethod
    def find(
            self,
            file_pattern: re.Pattern[str],
            base_dir: str | None = None,
            progress: ProgressReporter | None = None,
            file_filter: dict[str, Any] | None = None,
            max_count: int = -1,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        pass

    @abstractmethod
    async def get(
            self,
            key: str,
            as_bytes: bool | None = None,
            encoding: str | None = None,
    ) -> Any:
        pass

    @abstractmethod
    async def set(
            self,
            key: str,
            value: str | bytes | None,
            encoding: str | None = None,
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
    def child(
            self,
            name: str | None,
    ) -> "PipelineStorage":
        pass
