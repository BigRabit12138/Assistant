from typing import Any

from assistant.memory.graphrag_v1.index.cache.pipeline_cache import PipelineCache


class NoopPipelineCache(PipelineCache):
    async def get(self, key: str) -> Any:
        return None

    async def set(
            self,
            key: str,
            value: str | bytes | None,
            debug_data: dict | None = None
    ) -> None:
        pass

    async def has(self, key: str) -> bool:
        return False

    async def delete(self, key: str) -> None:
        pass

    async def clear(self) -> None:
        pass

    def child(self, name: str) -> PipelineCache:
        return self
