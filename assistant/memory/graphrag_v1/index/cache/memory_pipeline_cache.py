from typing import Any

from assistant.memory.graphrag_v1.index.cache.pipeline_cache import PipelineCache


class InMemoryCache(PipelineCache):
    _cache: dict[str, Any]
    _name: str

    def __init__(self, name: str | None = None):
        self._cache = {}
        self._name = name or ""

    async def get(self, key: str) -> Any:
        key = self._create_cache_key(key)
        return self._cache.get(key)

    async def set(
            self,
            key: str,
            value: Any,
            debug_data: dict | None = None
    ) -> None:
        key = self._create_cache_key(key)
        self._cache[key] = value

    async def has(self, key: str) -> bool:
        key = self._create_cache_key(key)
        return key in self._cache

    async def delete(self, key: str) -> None:
        key = self._create_cache_key(key)
        del self._cache[key]

    async def clear(self) -> None:
        self._cache.clear()

    def child(self, name: str) -> PipelineCache:
        return InMemoryCache(name)

    def _create_cache_key(self, key: str) -> str:
        return f"{self._name}{key}"


def create_memory_cache() -> PipelineCache:
    return InMemoryCache()
