from typing import Any
from dataclasses import dataclass
from collections.abc import Callable, Awaitable

from datashaper import VerbCallbacks

from assistant.memory.graphrag_v1.index.cache import PipelineCache

ExtractedEntity = dict[str, Any]
StrategyConfig = dict[str, Any]
EntityTypes = list[str]


@dataclass
class Document:
    text: str
    id: str


@dataclass
class EntityExtractionResult:
    """
    实体抽取结果
    """
    entities: list[ExtractedEntity]
    graphml_graph: str | None


EntityExtractStrategy = Callable[
    [
        list[Document],
        EntityTypes,
        VerbCallbacks,
        PipelineCache,
        StrategyConfig,
    ],
    Awaitable[EntityExtractionResult],
]
