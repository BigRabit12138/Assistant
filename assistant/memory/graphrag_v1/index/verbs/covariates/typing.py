from typing import Any
from dataclasses import dataclass
from collections.abc import (
    Awaitable,
    Callable,
    Iterable
)

from datashaper import VerbCallbacks

from assistant.memory.graphrag_v1.index.cache import PipelineCache


@dataclass
class Covariate:
    covariate_type: str | None = None
    subject_id: str | None = None
    subject_type: str | None = None
    object_id: str | None = None
    object_type: str | None = None
    type: str | None = None
    status: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: str | None = None
    source_text: list[str] | None = None
    doc_id: str | None = None
    record_id: int | None = None
    id: str | None = None


@dataclass
class CovariateExtractionResult:
    covariate_data: list[Covariate]


CovariateExtractStrategy = Callable[
    [
        Iterable[str],
        list[str],
        dict[str, str],
        VerbCallbacks,
        PipelineCache,
        dict[str, Any]
    ],
    Awaitable[CovariateExtractionResult],
]
