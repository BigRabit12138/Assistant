from dataclasses import dataclass
from collections.abc import Callable, Awaitable

from datashaper import VerbCallbacks

from assistant.memory.graphrag_v1.index.cache import PipelineCache


@dataclass
class TextEmbeddingResult:
    """
    embedding结果
    """
    embeddings: list[list[float] | None] | None


# embedding方法
TextEmbeddingStrategy = Callable[
    [
        list[str],
        VerbCallbacks,
        PipelineCache,
        dict,
    ],
    Awaitable[TextEmbeddingResult],
]
