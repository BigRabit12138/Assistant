from abc import ABC, abstractmethod

import pandas as pd

from assistant.memory.graphrag_v1.query.context_builder.conversation_history import ConversationHistory


class GlobalContextBuilder(ABC):
    @abstractmethod
    def build_context(
            self,
            conversation_history: ConversationHistory | None = None,
            **kwargs,
    ) -> tuple[str | list[str], dict[str, pd.DataFrame]]:
        pass


class LocalContextBuilder(ABC):
    @abstractmethod
    def build_context(
            self,
            query: str,
            conversation_history: ConversationHistory | None = None,
            **kwargs,
    ) -> tuple[str | list[str], dict[str, pd.DataFrame]]:
        pass
