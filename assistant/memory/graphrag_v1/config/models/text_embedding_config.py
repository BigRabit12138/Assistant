from pydantic import Field

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.config.enums import TextEmbeddingTarget
from assistant.memory.graphrag_v1.config.models.llm_config import LLMConfig


class TextEmbeddingConfig(LLMConfig):
    batch_size: int = Field(
        description="The batch size to use.",
        default=defaults.EMBEDDING_BATCH_SIZE,
    )
    batch_max_tokens: int = Field(
        description="The batch max tokens to use.",
        default=defaults.EMBEDDING_BATCH_MAX_TOKENS,
    )
    target: TextEmbeddingTarget = Field(
        description="The target to use. 'all' pr 'required'.",
        default=defaults.EMBEDDING_TARGET,
    )
    skip: list[str] = Field(
        description="The specific embeddings to skip.",
        default=[],
    )
    vector_store: dict | None = Field(
        description="The vector storage configuration.",
        default=None,
    )
    strategy: dict | None = Field(
        description="The override strategy to use.",
        default=None,
    )

    def resolved_strategy(self) -> dict:
        from assistant.memory.graphrag_v1.index.verbs.text.embed import TextEmbedStrategyType

        return self.strategy or {
            "type": TextEmbedStrategyType.openai,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "batch_size": self.batch_size,
            "batch_max_tokens": self.batch_max_tokens,
        }
