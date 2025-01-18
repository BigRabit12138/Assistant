from typing import Any
from collections.abc import Callable, Iterable

from datashaper import ProgressTicker

from assistant.memory.graphrag_v1.index.verbs.text.chunk.typing import TextChunk

ChunkStrategy = Callable[
    [list[str], dict[str, Any], ProgressTicker], Iterable[TextChunk]
]
