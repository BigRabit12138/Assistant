from typing import Protocol


class LLMConfig(Protocol):
    """
    大模型基础基类
    """
    @property
    def max_retries(self) -> int | None:
        pass

    @property
    def max_retry_wait(self) -> float | None:
        pass

    @property
    def sleep_on_rate_limit_recommendation(self) -> bool | None:
        pass

    @property
    def tokens_per_minute(self) -> int | None:
        pass

    @property
    def requests_per_minute(self) -> int | None:
        pass
