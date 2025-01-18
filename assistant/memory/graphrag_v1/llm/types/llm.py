from typing import (
    Generic,
    Protocol,
    TypeVar
)

from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.types.llm_io import (
    LLMInput,
    LLMOutput
)

TIn = TypeVar("TIn", contravariant=True)
TOut = TypeVar("TOut")


class LLM(Protocol, Generic[TIn, TOut]):
    async def __call__(
            self,
            input_: TIn,
            **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[TOut]:
        pass
