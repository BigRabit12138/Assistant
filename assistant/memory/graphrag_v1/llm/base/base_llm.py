import traceback

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.types import (
    LLM,
    LLMInput,
    LLMOutput,
    ErrorHandlerFn,
)

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class BaseLLM(ABC, LLM[TIn, TOut], Generic[TIn, TOut]):
    """
    大模型基类
    """
    _on_error: ErrorHandlerFn | None

    def on_error(self, on_error: ErrorHandlerFn | None) -> None:
        """
        设置错误处理函数
        :param on_error: 误处理函数
        :return:
        """
        self._on_error = on_error

    @abstractmethod
    async def _execute_llm(
            self,
            input_: TIn,
            **kwargs: Unpack[LLMInput],
    ) -> TOut | None:
        """
        子类实例化的调用模型方法
        :param input_: 模型输入
        :param kwargs: 模型参数
        :return: 模型输出
        """
        pass

    async def __call__(
            self, input_: TIn,
            **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[TOut]:
        """
        调用大模型
        :param input_: 模型输入
        :param kwargs: 模型参数
        :return: 模型输出
        """
        is_json = kwargs.get("json") or False
        if is_json:
            return await self._invoke_json(input_, **kwargs)
        return await self._invoke(input_, **kwargs)

    async def _invoke(
            self,
            input_: TIn,
            **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[TOut]:
        """
        调用大模型
        :param input_: 模型输入
        :param kwargs: 模型参数
        :return: 模型输出
        """
        try:
            output = await self._execute_llm(input_, **kwargs)
            return LLMOutput(output=output)
        except Exception as e:
            stack_trace = traceback.format_exc()
            if self._on_error:
                # 执行错误回调
                self._on_error(e, stack_trace, {"input_": input_})
            raise
        
    @abstractmethod
    async def _invoke_json(
            self,
            input_: TIn,
            **kwargs: Unpack[LLMInput]
    ) -> LLMOutput:
        """
        调用支持json格式输出的模型
        :param self: self
        :param input_: 模型输入
        :param kwargs: 模型参数
        :return: 模型输出
        """
        pass
