from typing_extensions import NotRequired

from assistant.memory.graphrag_v1.config.enums import TextEmbeddingTarget
from assistant.memory.graphrag_v1.config.input_models.llm_config_input import LLMConfigInput


class TextEmbeddingConfigInput(LLMConfigInput):
    batch_size: NotRequired[int | str | None]
    batch_max_tokens: NotRequired[int | str | None]
    target: NotRequired[TextEmbeddingTarget | str | None]
    skip: NotRequired[list[str] | str | None]
    vector_store: NotRequired[dict | None]
    strategy: NotRequired[dict | None]
