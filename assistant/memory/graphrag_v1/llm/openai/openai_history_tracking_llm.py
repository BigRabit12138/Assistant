from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.types import (
    LLM,
    LLMInput,
    LLMOutput,
    CompletionLLM,
    CompletionInput,
    CompletionOutput
)


class OpenAIHistoryTrackingLLM(LLM[CompletionInput, CompletionOutput]):
    """
    带有历史记忆的大模型
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
        调用大模型，并将输出添加到历史
        :param input_:
        :param kwargs:
        :return:
        """
        history = kwargs.get("history") or []
        output = await self._delegate(input_, **kwargs)
        return LLMOutput(
            output=output.output,
            json=output.json,
            history=[*history, {"role": "system", "content": output.output}],
        )
