from enum import Enum
from typing import Any, cast

import pandas as pd
import networkx as nx

from datashaper import (
    verb,
    VerbInput,
    VerbCallbacks,
    TableContainer,
    progress_callback,
)

from assistant.memory.graphrag_v1.index.utils import load_graph
from assistant.memory.graphrag_v1.index.graph.visualization import GraphLayout
from assistant.memory.graphrag_v1.index.verbs.graph.embed.typing import NodeEmbeddings


class LayoutGraphStrategyType(str, Enum):
    """
    图空间布局策略类型
    """
    umap = "umap"
    zero = "zero"

    def __repr__(self):
        return f'"{self.value}"'


@verb(name="layout_graph")
def layout_graph(
        input: VerbInput,
        callbacks: VerbCallbacks,
        strategy: dict[str, Any],
        embeddings_column: str,
        graph_column: str,
        to: str,
        graph_to: str | None = None,
        **_kwargs: dict,
) -> TableContainer:
    """
    获取图的节点位置布局
    :param input: 输入，包含输入表格
    :param callbacks: 回调钩子
    :param strategy: 布局策略
    :param embeddings_column: 图embedding列
    :param graph_column: 图所在列
    :param to: 节点位置信息新列
    :param graph_to: 在节点的位置信息的新图的新列
    :param _kwargs: 额外参数
    :return: 输出
    """
    output_df = cast(pd.DataFrame, input.get_input())

    num_items = len(output_df)
    strategy_type = strategy.get("type", LayoutGraphStrategyType.umap)
    strategy_args = {**strategy}

    has_embeddings = embeddings_column in output_df.columns

    # 获取每一个图的节点位置布局
    layouts = output_df.apply(
        progress_callback(
            lambda row: _run_layout(
                strategy_type,
                row[graph_column],
                row[embeddings_column] if has_embeddings else {},
                strategy_args,
                callbacks,
            ),
            callbacks.progress,
            num_items,
        ),
        axis=1,
    )
    # to列的每一个元素是[(key, 属性)]，属性是节点的位置信息和簇度信息
    output_df[to] = layouts.apply(lambda layout: [pos.to_pandas() for pos in layout])
    if graph_to is not None:
        # 对每一个图中的节点增加位置信息，赋值给graph_to列
        output_df[graph_to] = output_df.apply(
            lambda row: _apply_layout_to_graph(
                row[graph_column], cast(GraphLayout, layouts[row.name])
            ),
            axis=1,
        )
    return TableContainer(table=output_df)


def _run_layout(
        strategy: LayoutGraphStrategyType,
        graphml_or_graph: str | nx.Graph,
        embeddings: NodeEmbeddings,
        args: dict[str, Any],
        reporter: VerbCallbacks,
) -> GraphLayout:
    """
    计算一个图的节点的空间布局
    :param strategy: 布局策略
    :param graphml_or_graph: networkx图
    :param embeddings: 图embedding
    :param args: 策略参数
    :param reporter: 回调钩子
    :return: 图的空间布局
    """
    # 加载图
    graph = load_graph(graphml_or_graph)
    match strategy:
        case LayoutGraphStrategyType.umap:
            from assistant.memory.graphrag_v1.index.verbs.graph.layout.methods.umap import run as run_umap

            return run_umap(
                graph,
                embeddings,
                args,
                lambda e, stack, d: reporter.error("Error in Umap", e, stack, d),
            )
        case LayoutGraphStrategyType.zero:
            from assistant.memory.graphrag_v1.index.verbs.graph.layout.methods.zero import run as run_zero
            # 将节点放置在2维原点
            return run_zero(
                graph,
                args,
                lambda e, stack, d: reporter.error("Error in Zero", d, stack, d),
            )
        case _:
            msg = f"Unknown strategy {strategy}."
            raise ValueError(msg)


def _apply_layout_to_graph(
        graphml_or_graph: str | nx.Graph,
        layout: GraphLayout
) -> str:
    """
    将图节点的布局信息添加到图节点的属性里面去，只有x，y
    :param graphml_or_graph: networkx图
    :param layout: 图节点布局
    :return: 带位置信息的图
    """
    # 加载图
    graph = load_graph(graphml_or_graph)
    for node_position in layout:
        if node_position.label in graph.nodes:
            # 节点增加位置信息
            graph.nodes[node_position.label]["x"] = node_position.x
            graph.nodes[node_position.label]["y"] = node_position.y
            graph.nodes[node_position.label]["size"] = node_position.size

    return "\n".join(nx.generate_graphml(graph))
