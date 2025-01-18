from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.openai.utils import perform_variable_replacements
from assistant.memory.graphrag_v1.llm.types import (
    LLM,
    LLMInput,
    LLMOutput,
    CompletionLLM,
    CompletionInput,
    CompletionOutput,
)


class OpenAITokenReplacingLLM(LLM[CompletionInput, CompletionOutput]):
    """
    带有模板变量实例化的模型
    """
    _delegate: CompletionLLM

    def __init__(self, delegate: CompletionLLM):
        self._delegate = delegate

    async def __call__(
            self,
            input_: CompletionInput,
            **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        """
        实例化模板，并调用大模型
        :param input_: 模板
        :param kwargs: 变量
        :return: 大模型输出
        """
        variables = kwargs.get("variables")
        history = kwargs.get("history") or []
        input_ = perform_variable_replacements(input_, history, variables)
        return await self._delegate(input_, **kwargs)
