import logging

from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.base import BaseLLM
from assistant.memory.graphrag_v1.llm.base.base_llm import TIn, TOut
from assistant.memory.graphrag_v1.llm.types import (
    LLMInput,
    CompletionInput,
    CompletionOutput,
)

log = logging.getLogger(__name__)


class MockCompletionLLM(
    BaseLLM[
        CompletionInput,
        CompletionOutput
    ]
):
    def __init__(
            self,
            responses: list[str],
    ):
        self.responses = responses
        self._on_error = None

    async def _execute_llm(
            self,
            input_: CompletionInput,
            **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput:
        return self.responses[0]
