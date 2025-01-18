from typing import Any
from collections.abc import Awaitable, Callable

from datashaper import VerbCallbacks
from typing_extensions import TypedDict

from assistant.memory.graphrag_v1.index.cache import PipelineCache

ExtractedEntity = dict[str, Any]
StrategyConfig = dict[str, Any]
RowContext = dict[str, Any]
EntityTypes = list[str]
Claim = dict[str, Any]


class Finding(TypedDict):
    summary: str
    explanation: str


class CommunityReport(TypedDict):
    """
    聚簇的总结
    """
    community: str | int
    title: str
    summary: str
    full_content: str
    full_content_json: str
    rank: float
    level: int
    rank_explanation: str
    findings: list[Finding]


CommunityReportsStrategy = Callable[
    [
        str | int,
        str,
        int,
        VerbCallbacks,
        PipelineCache,
        StrategyConfig,
    ],
    Awaitable[CommunityReport | None],
]
