from typing import Any

from assistant.memory.graphrag_v1.index.storage.typing import PipelineStorage
from assistant.memory.graphrag_v1.index.storage.file_pipline_storage import FilePipelineStorage


class MemoryPiplineStorage(FilePipelineStorage):
    _storage: dict[str, Any]

    def __init__(self):
        super().__init__(root_dir=".output")
        self._storage = {}

    async def get(
            self,
            key: str,
            as_bytes: bool | None = None,
            encoding: str | None = None,
    ) -> Any:
        return self._storage.get(key) or await super().get(
            key,
            as_bytes,
            encoding
        )

    async def set(
            self,
            key: str,
            value: str | bytes | None,
            encoding: str | None = None,
    ) -> None:
        self._storage[key] = value

    async def has(self, key: str) -> bool:
        return key in self._storage or await super().has(key)

    async def delete(self, key: str) -> None:
        del self._storage[key]

    async def clear(self) -> None:
        self._storage.clear()

    def child(
            self,
            name: str | None,
    ) -> "PipelineStorage":
        return self


def create_memory_storage() -> PipelineStorage:
    return MemoryPiplineStorage()
