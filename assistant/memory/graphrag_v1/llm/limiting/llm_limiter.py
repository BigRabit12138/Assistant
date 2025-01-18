from abc import ABC, abstractmethod


class LLMLimiter(ABC):
    """
    模型限制器抽象基类
    """
    @property
    @abstractmethod
    def needs_token_count(self) -> bool:
        """
        模型是否需要统计Tokens用量
        :return: 是否需要
        """
        pass

    @abstractmethod
    async def acquire(self, num_tokens: int = 1) -> None:
        """
        获取锁
        :param num_tokens: tokens用量
        :return:
        """
        pass
