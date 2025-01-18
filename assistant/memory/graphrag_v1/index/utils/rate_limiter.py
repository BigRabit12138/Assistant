import time
import asyncio


class RateLimiter:
    """
    速率限制器
    """
    def __init__(
            self,
            rate: int,
            per: int
    ):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.monotonic()

    async def acquire(self):
        """
        限制速率
        :return:
        """
        current = time.monotonic()
        elapsed = current - self.last_check
        self.last_check = current
        self.allowance += elapsed * (self.rate / self.per)

        if self.allowance > self.rate:
            self.allowance = self.rate

        if self.allowance < 1.0:
            sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
            await asyncio.sleep(sleep_time)
            self.allowance = 0.0
        else:
            self.allowance -= 1.0
