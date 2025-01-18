from typing_extensions import NotRequired

from assistant.memory.graphrag_v1.config.input_models.llm_config_input import LLMConfigInput


class EntityExtractionConfigInput(LLMConfigInput):
    prompt: NotRequired[str | None]
    entity_types: NotRequired[list[str] | str | None]
    max_gleanings: NotRequired[int | str | None]
    strategy: NotRequired[dict | None]
