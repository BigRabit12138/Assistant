from assistant.memory.graphrag_v1.llm.limiting.llm_limiter import LLMLimiter


class CompositeLLMLimiter(LLMLimiter):
    _limiters: list[LLMLimiter]

    def __init__(self, limiters: list[LLMLimiter]):
        self._limiters = limiters

    @property
    def needs_token_count(self) -> bool:
        return any(limiter.needs_token_count for limiter in self._limiters)

    async def acquire(self, num_tokens: int = 1) -> None:
        for limiter in self._limiters:
            await limiter.acquire(num_tokens)
