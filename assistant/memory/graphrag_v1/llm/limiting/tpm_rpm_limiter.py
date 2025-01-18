from aiolimiter import AsyncLimiter

from assistant.memory.graphrag_v1.llm.limiting.llm_limiter import LLMLimiter


class TpmRpmLLMLimiter(LLMLimiter):
    """
    模型速率限制器
    """
    _tpm_limiter: AsyncLimiter | None
    _rpm_limiter: AsyncLimiter | None

    def __init__(
            self,
            tpm_limiter: AsyncLimiter | None,
            rpm_limiter: AsyncLimiter | None
    ):
        self._tpm_limiter = tpm_limiter
        self._rpm_limiter = rpm_limiter

    @property
    def needs_token_count(self) -> bool:
        """
        是否需要考虑token速率的限制
        :return: 是否需要考虑token速率的限制
        """
        return self._tpm_limiter is not None

    async def acquire(self, num_tokens: int = 1) -> None:
        """
        获取可执行权限，如果达到容量限制，等待资源满足条件
        :param num_tokens:
        :return:
        """
        if self._tpm_limiter is not None:
            await self._tpm_limiter.acquire(num_tokens)
        if self._rpm_limiter is not None:
            await self._rpm_limiter.acquire()
