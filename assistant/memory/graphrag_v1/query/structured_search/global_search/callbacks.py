from assistant.memory.graphrag_v1.query.llm.base import BaseLLMCallback
from assistant.memory.graphrag_v1.query.structured_search.base import SearchResult


class GlobalSearchLLMCallback(BaseLLMCallback):
    def __init__(self):
        super().__init__()
        self.map_response_contexts = []
        self.map_response_outputs = []

    def on_map_response_start(
            self,
            map_response_contexts: list[str],
    ):
        self.map_response_contexts = map_response_contexts

    def on_map_response_end(
            self,
            map_response_outputs: list[SearchResult],
    ):
        self.map_response_outputs = map_response_outputs
