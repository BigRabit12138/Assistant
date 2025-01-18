from typing import Generic, TypeVar
from dataclasses import dataclass, field

from typing_extensions import NotRequired, TypedDict

from assistant.memory.graphrag_v1.llm.types.llm_callbacks import IsResponseValidFn


class LLMInput(TypedDict):
    """
    大模型输入
    """
    name: NotRequired[str]

    json: NotRequired[bool]

    is_response_valid: NotRequired[IsResponseValidFn]

    variables: NotRequired[dict]

    history: NotRequired[list[dict]]

    model_parameters: NotRequired[dict]


T = TypeVar("T")


@dataclass
class LLMOutput(Generic[T]):
    """
    大模型输出结果
    """
    output: T | None

    json: dict | None = field(default=None)

    history: list[dict] | None = field(default=None)
