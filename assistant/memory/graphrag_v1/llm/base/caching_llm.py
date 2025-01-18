import json

from typing import Any, Generic, TypeVar

from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.base._create_cache_key import create_hash_key
from assistant.memory.graphrag_v1.llm.types import (
    LLM,
    LLMCache,
    LLMInput,
    LLMOutput,
    OnCacheActionFn,
)

_cache_strategy_version = 2

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


def _noop_cache_fn(_k: str, _v: str | None):
    pass


class CacheingLLM(LLM[TIn, TOut], Generic[TIn, TOut]):
    """
    带有缓存器的大模型
    """
    _cache: LLMCache
    _delegate: LLM[TIn, TOut]
    _operation: str
    _llm_parameters: dict
    _on_cache_hit: OnCacheActionFn
    _on_cache_miss: OnCacheActionFn

    def __init__(
            self,
            delegate: LLM[TIn, TOut],
            llm_parameters: dict,
            operation: str,
            cache: LLMCache,
    ):
        self._delegate = delegate
        self._llm_parameters = llm_parameters
        self._cache = cache
        self._operation = operation
        self._on_cache_hit = _noop_cache_fn
        self._on_cache_miss = _noop_cache_fn

    def on_cache_hit(self, fn: OnCacheActionFn | None) -> None:
        """
        设置找到缓存回调函数
        :param fn: 回调函数
        :return:
        """
        self._on_cache_hit = fn or _noop_cache_fn

    def on_cache_miss(self, fn: OnCacheActionFn | None) -> None:
        """
        设置缺失缓存回调函数
        :param fn: 回调函数
        :return:
        """
        self._on_cache_miss = fn or _noop_cache_fn

    def _cache_key(
            self,
            input_: TIn,
            name: str | None,
            args: dict
    ) -> str:
        """
        创建缓存标识ID
        :param input_: 大模型输入
        :param name: 名字
        :param args: 模型参数
        :return:
        """
        json_input = json.dumps(input_)
        tag = (
            f"{name}-{self._operation}-v{_cache_strategy_version}"
            if name is not None
            else self._operation
        )
        return create_hash_key(tag, json_input, args)

    async def _cache_read(self, key: str) -> Any | None:
        """
        从缓存器中获取历史结果
        :param key: 缓存ID
        :return: 缓存结果
        """
        return await self._cache.get(key)

    async def _cache_write(
            self,
            key: str,
            input_: TIn,
            result: TOut | None,
            args: dict
    ) -> None:
        """
        添加新缓存结果
        :param key: 缓存ID
        :param input_: 大模型输入
        :param result: 大模型输出
        :param args: 参数
        :return:
        """
        if result:
            await self._cache.set(
                key,
                result,
                {
                    "input_": input_,
                    "parameters": args
                }
            )

    async def __call__(
            self,
            input_: TIn,
            **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[TOut]:
        name = kwargs.get("name")
        llm_args = {**self._llm_parameters, **(kwargs.get("model_parameters") or {})}
        # 获取缓存标识ID
        cache_key = self._cache_key(input_, name, llm_args)
        # 获取缓存结果
        cached_result = await self._cache_read(cache_key)
        if cached_result:
            self._on_cache_hit(cache_key, name)
            return LLMOutput(output=cached_result)

        self._on_cache_miss(cache_key, name)

        result = await self._delegate(input_, **kwargs)
        # 添加缓存
        await self._cache_write(
            cache_key,
            input_,
            result.output,
            llm_args
        )
        return result
