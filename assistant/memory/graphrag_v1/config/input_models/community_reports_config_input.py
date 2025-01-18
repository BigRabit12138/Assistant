from typing_extensions import NotRequired

from assistant.memory.graphrag_v1.config.input_models.llm_config_input import LLMConfigInput


class CommunityReportsConfigInput(LLMConfigInput):
    prompt: NotRequired[str | None]
    max_length: NotRequired[int | str | None]
    max_input_length: NotRequired[int | str | None]
    strategy: NotRequired[dict | None]
