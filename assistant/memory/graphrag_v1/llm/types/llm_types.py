from typing import TypeAlias

from assistant.memory.graphrag_v1.llm.types.llm import LLM

# embedding输入
EmbeddingInput: TypeAlias = list[str]
# embedding输出
EmbeddingOutput: TypeAlias = list[list[float]]
CompletionInput: TypeAlias = str
CompletionOutput: TypeAlias = str

EmbeddingLLM: TypeAlias = LLM[EmbeddingInput, EmbeddingOutput]
CompletionLLM: TypeAlias = LLM[CompletionInput, CompletionOutput]
