from typing_extensions import NotRequired

from assistant.memory.graphrag_v1.config.input_models.llm_config_input import LLMConfigInput


class ClaimExtractionConfigInput(LLMConfigInput):
    enabled: NotRequired[bool | None]
    prompt: NotRequired[str | None]
    description: NotRequired[str | None]
    max_gleanings: NotRequired[int | str | None]
    strategy: NotRequired[dict | None]
