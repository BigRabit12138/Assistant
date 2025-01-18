from pathlib import Path
from pydantic import Field

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.config.models.llm_config import LLMConfig


class SummarizeDescriptionConfig(LLMConfig):
    prompt: str | None = Field(
        description="The description summarization prompt to use.",
        default=None,
    )
    max_length: int = Field(
        description="The description summarization maximum length.",
        default=defaults.SUMMARIZE_DESCRIPTIONS_MAX_LENGTH,
    )
    strategy: dict | None = Field(
        description="The override strategy to use.",
        default=None,
    )

    def resolved_strategy(self, root_dir: str) -> dict:
        from assistant.memory.graphrag_v1.index.verbs.entities.summarize import SummarizeStrategyType

        return self.strategy or {
            "type": SummarizeStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "summarize_prompt": (Path(root_dir) / self.prompt)
            .read_bytes()
            .decode(encoding="utf-8")
            if self.prompt
            else None,
            "max_summary_length": self.max_length,
        }
