import random

from typing import Any
from collections.abc import Iterable

from datashaper import (
    VerbCallbacks,
    ProgressTicker,
    progress_ticker,
)

from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.verbs.text.embed.strategies.typing import TextEmbeddingResult


async def run(
        input_: list[str],
        callbacks: VerbCallbacks,
        cache: PipelineCache,
        _args: dict[str, Any],
) -> TextEmbeddingResult:
    input_ = input_ if isinstance(input_, Iterable) else [input_]
    ticker = progress_ticker(callbacks.progress, len(input_))
    return TextEmbeddingResult(
        embeddings=[_embed_text(cache, text, ticker) for text in input_]
    )


def _embed_text(
        _cache: PipelineCache,
        _text: str,
        tick: ProgressTicker
) -> list[float]:
    tick(1)
    return [random.random(), random.random(), random.random()]
