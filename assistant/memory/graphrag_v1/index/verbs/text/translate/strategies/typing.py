from typing import Any
from dataclasses import dataclass
from collections.abc import Callable, Awaitable

from datashaper import VerbCallbacks

from assistant.memory.graphrag_v1.index.cache import PipelineCache


@dataclass
class TextTranslationResult:
    translations: list[str]


TextTranslationStrategy = Callable[
    [list[str], dict[str, Any], VerbCallbacks, PipelineCache],
    Awaitable[TextTranslationResult],
]
