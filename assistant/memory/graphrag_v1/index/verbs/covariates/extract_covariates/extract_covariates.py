import logging

from enum import Enum
from typing import Any, cast
from dataclasses import asdict

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    AsyncType,
    VerbCallbacks,
    TableContainer,
    derive_from_rows,
)

from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.verbs.covariates.typing import (
    Covariate,
    CovariateExtractStrategy
)

log = logging.getLogger(__name__)


class ExtractClaimsStrategyType(str, Enum):
    graph_intelligence = "graph_intelligence"

    def __repr__(self):
        return f'"{self.value}"'


DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


@verb(name="extract_covariates")
async def extract_covariates(
        input: VerbInput,
        cache: PipelineCache,
        callbacks: VerbCallbacks,
        column: str,
        covariate_type: str,
        strategy: dict[str, Any] | None,
        async_mode: AsyncType = AsyncType.AsyncIO,
        entity_types: list[str] | None = None,
        **kwargs,
) -> TableContainer:
    log.debug(f"extract_covariates strategy={strategy}")
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    output = cast(pd.DataFrame, input.get_input())

    resolved_entities_map = {}

    strategy = strategy or {}
    strategy_exec = load_strategy(
        strategy.get("type", ExtractClaimsStrategyType.graph_intelligence)
    )
    strategy_config = {**strategy}

    async def run_strategy(row):
        text = row[column]
        result = await strategy_exec(
            text,
            entity_types,
            resolved_entities_map,
            callbacks,
            cache,
            strategy_config
        )
        return [
            create_row_from_claim_data(row, item, covariate_type)
            for item in result.covariate_data
        ]

    results = await derive_from_rows(
        output,
        run_strategy,
        callbacks,
        scheduling_type=async_mode,
        num_threads=kwargs.get("num_threads", 4)
    )
    output = pd.DataFrame([item for row in results for item in row or []])
    return TableContainer(table=output)


def load_strategy(
        strategy_type: ExtractClaimsStrategyType
) -> CovariateExtractStrategy:
    match strategy_type:
        case ExtractClaimsStrategyType.graph_intelligence:
            from assistant.memory.graphrag_v1.index.verbs.covariates.extract_covariates.\
                strategies.graph_intelligence import run as run_gi

            return run_gi
        case _:
            msg = f"Unknown strategy: {strategy_type}."
            raise ValueError(msg)


def create_row_from_claim_data(
        row,
        covariate_data: Covariate,
        covariate_type: str
):
    item = {**row, **asdict(covariate_data), "covariate_type": covariate_type}
    del item["doc_id"]
    return item
