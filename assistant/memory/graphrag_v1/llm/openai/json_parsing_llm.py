from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.types import (
    LLM,
    LLMInput,
    LLMOutput,
    CompletionLLM,
    CompletionInput,
    CompletionOutput
)
from assistant.memory.graphrag_v1.llm.openai.utils import try_parse_json_object


class JsonParsingLLM(LLM[CompletionInput, CompletionOutput]):
    """
    带有JSON输出解析的模型
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
        获取大模型输出，并尝试解析JSON
        :param input_: 模板
        :param kwargs: 变量
        :return: 大模型输出
        """
        result = await self._delegate(input_, **kwargs)
        if kwargs.get("json") and result.json is None and result.output is not None:
            result.json = try_parse_json_object(result.output)
        return result
