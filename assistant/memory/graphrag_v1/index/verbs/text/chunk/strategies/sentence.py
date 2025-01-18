from typing import Any
from collections.abc import Iterable

import nltk

from datashaper import ProgressTicker

from assistant.memory.graphrag_v1.index.verbs.text.chunk.strategies.typing import TextChunk


def run(
        input_: list[str],
        _args: dict[str, Any],
        tick: ProgressTicker
) -> Iterable[TextChunk]:
    for doc_idx, text in enumerate(input_):
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            yield TextChunk(
                text_chunk=sentence,
                source_doc_indices=[doc_idx],
            )
        tick(1)
