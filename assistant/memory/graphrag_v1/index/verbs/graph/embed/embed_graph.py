from enum import Enum
from typing import Any, cast

import pandas as pd
import networkx as nx

from datashaper import (
    verb,
    VerbInput,
    VerbCallbacks,
    TableContainer,
    derive_from_rows,
)

from assistant.memory.graphrag_v1.index.utils import load_graph
from assistant.memory.graphrag_v1.index.verbs.graph.embed.typing import NodeEmbeddings


class EmbedGraphStrategyType(str, Enum):
    node2vec = "node2vec"

    def __repr__(self):
        return f'"{self.value}"'


@verb(name="embed_graph")
async def embed_graph(
        input: VerbInput,
        callbacks: VerbCallbacks,
        strategy: dict[str, Any],
        column: str,
        to: str,
        **kwargs,
) -> TableContainer:
    output_df = cast(pd.DataFrame, input.get_input())
    strategy_type = strategy.get("type", EmbedGraphStrategyType.node2vec)
    strategy_args = {**strategy}

    async def run_strategy(row):
        return run_embeddings(strategy_type, cast(Any, row[column]), strategy_args)

    results = await derive_from_rows(
        output_df,
        run_strategy,
        callbacks=callbacks,
        num_threads=kwargs.get("num_threads", 4),
    )

    output_df[to] = list(results)
    return TableContainer(table=output_df)


def run_embeddings(
        strategy: EmbedGraphStrategyType,
        graphml_or_graph: str | nx.Graph,
        args: dict[str, Any],
) -> NodeEmbeddings:
    graph = load_graph(graphml_or_graph)
    match strategy:
        case EmbedGraphStrategyType.node2vec:
            from assistant.memory.graphrag_v1.index.verbs.graph.embed.strategies.node_2_vec import run as run_node_2_vec

            return run_node_2_vec(graph, args)
        case _:
            msg = f"Unknown strategy {strategy}."
            raise ValueError(msg)
