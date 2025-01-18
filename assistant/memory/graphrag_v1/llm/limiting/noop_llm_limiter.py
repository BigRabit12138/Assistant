from assistant.memory.graphrag_v1.llm.limiting.llm_limiter import LLMLimiter


class NoopLLMLimiter(LLMLimiter):
    @property
    def needs_token_count(self) -> bool:
        return False

    async def acquire(self, num_tokens: int = 1) -> None:
        pass
