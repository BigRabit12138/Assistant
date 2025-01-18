from typing import Any, Protocol


class LLMCache(Protocol):
    async def has(self, key: str) -> bool:
        pass

    async def get(self, key: str) -> Any | None:
        pass

    async def set(
            self,
            key: str,
            value: Any,
            debug_data: dict | None = None
    ) -> None:
        pass
