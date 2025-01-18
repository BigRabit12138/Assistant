from assistant.memory.graphrag_v1.config.enums import LLMType

MOCK_LLM_RESPONSES = [
    """
    This is a MOCK response for the LLM. It is summarized!
    """.strip()
]

DEFAULT_LLM_CONFIG = {
    "type": LLMType.StaticResponse,
    "responses": MOCK_LLM_RESPONSES,
}
