from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class LLMInvocationResult(Generic[T]):
    """
    大模型输出结果
    """
    result: T | None
    name: str
    num_retries: int
    total_time: float
    call_times: list[float]
    input_tokens: int
    output_tokens: int
