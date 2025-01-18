from typing import Any, cast

import pandas as pd
import networkx as nx

from datashaper import (
    verb,
    VerbInput,
    VerbCallbacks,
    TableContainer,
    progress_iterable,
)

from assistant.memory.graphrag_v1.index.utils import load_graph

default_copy = ['level']


@verb(name="unpack_graph")
def unpack_graph(
        input: VerbInput,
        callbacks: VerbCallbacks,
        column: str,
        type: str,
        copy: list[str] | None = None,
        embeddings_column: str = "embeddings",
        **kwargs,
) -> TableContainer:
    """
    将图对象的数据解包，每个属性构成一列，加上复制的列和embeddings列
    :param input: 输入，包含输入表格
    :param callbacks: 回调钩子
    :param column: 图所在列
    :param type: 图对象类型，nodes或edges
    :param copy: 需要保留的列
    :param embeddings_column: embeddings所在列
    :param kwargs: 额外参数
    :return: 输出，包含输出表格
    """
    if copy is None:
        copy = default_copy
    input_df = input.get_input()
    num_total = len(input_df)
    result = []
    # 保留需要复制列中存在的列
    copy = [col for col in copy if col in input_df.columns]
    has_embeddings = embeddings_column in input_df.columns

    for _, row in progress_iterable(input_df.iterrows(), callbacks.progress, num_total):
        # 保留level列的内容
        cleaned_row = {col: row[col] for col in copy}
        embeddings = (
            cast(dict[str, list[float]], row[embeddings_column])
            if has_embeddings
            else {}
        )
        # 解包一个图对象的所有数据，和保留的列构成新的一行数据
        result.extend(
            [
                {**cleaned_row, **graph_id}
                for graph_id in _run_unpack(
                    cast(str | nx.Graph, row[column]),
                    type,
                    embeddings,
                    kwargs,
                )
            ]
        )

    output_df = pd.DataFrame(result)
    return TableContainer(table=output_df)


def _run_unpack(
        graphml_or_graph: str | nx.Graph,
        unpack_type: str,
        embeddings: dict[str, list[float]],
        args: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    获取图对象数据
    :param graphml_or_graph: networkx图
    :param unpack_type: 图对象类型，nodes或edges
    :param embeddings: 图embedding
    :param args: 额外参数
    :return: 图对象数据
    """
    graph = load_graph(graphml_or_graph)
    if unpack_type == "nodes":
        return _unpack_nodes(graph, embeddings, args)
    if unpack_type == "edges":
        return _unpack_edges(graph, args)
    msg = f"Unknown type {unpack_type}."
    raise ValueError(msg)


def _unpack_nodes(
        graph: nx.Graph,
        embeddings: dict[str, list[float]],
        _args: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    获取节点的数据
    :param graph: networkx图
    :param embeddings: 图embedding
    :param _args: 额外参数
    :return: 图节点数据
    """
    return [
        {
            "label": label,
            **(node_data or {}),
            "graph_embedding": embeddings.get(label),
        }
        for label, node_data in graph.nodes(data=True)
    ]


def _unpack_edges(
        graph: nx.Graph, _args: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    获取边节点数据
    :param graph: networkx图
    :param _args: 图embedding
    :return: 图边数据
    """
    return [
        {
            "source": source_id,
            "target": target_id,
            **(edge_data or {}),
        }
        for source_id, target_id, edge_data in graph.edges(data=True)
    ]
