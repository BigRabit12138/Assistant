from typing import Any
from dataclasses import dataclass
from collections.abc import Awaitable, Callable

from datashaper import VerbCallbacks

from assistant.memory.graphrag_v1.index.cache import PipelineCache

StrategyConfig = dict[str, Any]


@dataclass
class SummarizedDescriptionResult:
    """
    摘要结果
    """
    items: str | tuple[str, str]
    description: str


# 摘要策略方法
SummarizationStrategy = Callable[
    [
        str | tuple[str, str],
        list[str],
        VerbCallbacks,
        PipelineCache,
        StrategyConfig,
    ],
    Awaitable[SummarizedDescriptionResult],
]
