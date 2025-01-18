import json

from typing import Any

from assistant.memory.graphrag_v1.index.storage import PipelineStorage
from assistant.memory.graphrag_v1.index.cache.pipeline_cache import PipelineCache


class JsonPipelineCache(PipelineCache):
    _storage: PipelineStorage
    _encoding: str

    def __init__(
            self,
            storage: PipelineStorage,
            encoding="utf-8"
    ):
        self._storage = storage
        self._encoding = encoding

    async def get(self, key: str) -> str | None:
        if await self.has(key):
            try:
                data = await self._storage.get(key, encoding=self._encoding)
                data = json.loads(data)
            except UnicodeDecodeError:
                await self._storage.delete(key)
                return None
            except json.decoder.JSONDecodeError:
                await self._storage.delete(key)
                return None
            else:
                return data.get("result")

        return None

    async def set(
            self,
            key: str,
            value: Any,
            debug_data: dict | None = None
    ) -> None:
        if value is None:
            return
        data = {"result": value, **(debug_data or {})}
        await self._storage.set(key, json.dumps(data), encoding=self._encoding)

    async def has(self, key: str) -> bool:
        return await self._storage.has(key)

    async def delete(self, key: str) -> None:
        if await self.has(key):
            await self._storage.delete(key)

    async def clear(self) -> None:
        await self._storage.clear()

    def child(self, name: str) -> PipelineCache:
        return JsonPipelineCache(self._storage.child(name), encoding=self._encoding)
