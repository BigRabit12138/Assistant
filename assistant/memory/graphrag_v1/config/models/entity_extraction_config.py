from pathlib import Path
from pydantic import Field

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.config.models.llm_config import LLMConfig


class EntityExtractionConfig(LLMConfig):
    prompt: str | None = Field(
        description="The entity extraction prompt to use.",
        default=None,
    )
    entity_types: list[str] = Field(
        description="The entity extraction entity types to use.",
        default=defaults.ENTITY_EXTRACTION_ENTITY_TYPES,
    )
    max_gleanings: int = Field(
        description="The maximum number of entity gleanings to use.",
        default=defaults.ENTITY_EXTRACTION_MAX_GLEANINGS,
    )
    strategy: dict | None = Field(
        description="Override the default entity extraction strategy.",
        default=None,
    )
    encoding_model: str | None = Field(
        default=None, description="The encoding model to use."
    )

    def resolved_strategy(
            self,
            root_dir: str,
            encoding_model: str
    ) -> dict:
        from assistant.memory.graphrag_v1.index.verbs.entities.extraction import ExtractEntityStrategyType

        return self.strategy or {
            "type": ExtractEntityStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "extraction_prompt": (Path(root_dir) / self.prompt)
            .read_bytes()
            .decode(encoding="utf-8")
            if self.prompt
            else None,
            "max_gleanings": self.max_gleanings,
            "encoding_name": self.encoding_model or encoding_model,
            "prechunked": True,
        }
