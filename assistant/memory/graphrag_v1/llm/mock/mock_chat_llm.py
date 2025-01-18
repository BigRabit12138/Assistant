from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.base import BaseLLM
from assistant.memory.graphrag_v1.llm.base.base_llm import TIn, TOut
from assistant.memory.graphrag_v1.llm.types import (
    LLMInput,
    LLMOutput,
    CompletionInput,
    CompletionOutput,
)


class MockChatLLM(
    BaseLLM[
        CompletionInput,
        CompletionOutput,
    ]
):
    responses: list[str]
    i: int = 0

    def __init__(
            self,
            responses: list[str]
    ):
        self.i = 0
        self.responses = responses

    @staticmethod
    def _create_output(
            self,
            output: CompletionOutput | None,
            **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[CompletionOutput]:
        history = kwargs.get("history") or []
        return LLMOutput[CompletionOutput](
            output=output, history=[*history, {"content": output}]
        )

    async def _execute_llm(
            self,
            input_: CompletionInput,
            **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput:
        if self.i >= len(self.responses):
            msg = f"No more responses, requested {self.i} but only have {len(self.responses)}."
            raise ValueError(msg)

        response = self.responses[self.i]
        self.i += 1
        return response
