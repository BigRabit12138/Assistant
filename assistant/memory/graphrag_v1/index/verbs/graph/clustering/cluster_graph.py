import logging

from enum import Enum
from random import Random
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
from assistant.memory.graphrag_v1.index.verbs.graph.clustering.typing import Communities
from assistant.memory.graphrag_v1.index.utils import (
    gen_uuid,
    load_graph,
)

log = logging.getLogger(__name__)


@verb(name="cluster_graph")
def cluster_graph(
        input: VerbInput,
        callbacks: VerbCallbacks,
        strategy: dict[str, Any],
        column: str,
        to: str,
        level_to: str | None = None,
        **kwargs,
) -> TableContainer:
    """
    对图进行聚类，将每一层的结果和更新了这一层信息的图赋值to和level_to列
    :param input: 输入，包含输入表格
    :param callbacks: 进度条
    :param strategy: 聚类策略
    :param column: 图所在列
    :param to: 某一层的聚类后的图
    :param level_to: 某一层的聚类后的图
    :param kwargs: 额外的参数
    :return: 输出包含输出表格
    """
    output_df = cast(pd.DataFrame, input.get_input())
    # results: [[(层次, 聚族ID, [节点ID])]]
    results = output_df[column].apply(lambda graph_: run_layout(strategy, graph_))

    # 将聚簇结果添加到communities列
    community_map_to = "communities"
    output_df[community_map_to] = results

    level_to = level_to or f"{to}_level"
    # 获取所有层次，并去重（去重没用），添加到level列
    output_df[level_to] = output_df.apply(
        lambda x: list({level_ for level_, _, _ in x[community_map_to]}), axis=1
    )

    output_df[to] = [None] * len(output_df)
    num_total = len(output_df)

    # graph_level_pairs_column: [[(层次, 更新了本层节点的聚类属性的图)]]
    graph_level_pairs_column: list[list[tuple[int, str]]] = []
    for _, row in progress_iterable(
        output_df.iterrows(), callbacks.progress, num_total
    ):
        levels = row[level_to]
        # graph_level_pairs: [(层次, 更新了本层节点的聚类属性的图)]
        graph_level_pairs: list[tuple[int, str]] = []

        # 每个层次更新一个图
        for level in levels:
            graph = "\n".join(
                nx.generate_graphml(
                    apply_clustering(
                        cast(str, row[column]),
                        cast(Communities, row[community_map_to]),
                        level,
                    )
                )
            )
            graph_level_pairs.append((level, graph))
        graph_level_pairs_column.append(graph_level_pairs)
    # 每一行的clustered_graph列是[(层次, 更新了本层节点的聚类属性的图)]
    output_df[to] = graph_level_pairs_column
    # 将clustered_graph列展开，并重置索引
    output_df = output_df.explode(to, ignore_index=True)
    # 将to(clustered_graph)列的内容展开，赋给level_to, to
    # level_to内容为层次, to内容为更新了本层节点的聚类属性的图
    output_df[[level_to, to]] = pd.DataFrame(
        output_df[to].tolist(), index=output_df.index
    )

    output_df.drop(columns=[community_map_to], inplace=True)

    return TableContainer(table=output_df)


class GraphCommunityStrategyType(str, Enum):
    """
    图聚类算法
    """
    leiden = "leiden"

    def __repr__(self):
        return f'"{self.value}"'


def apply_clustering(
        graphml: str,
        communities: Communities,
        level=0,
        seed=0xF001
) -> nx.Graph:
    """
    对graphml图中被分为level层的节点设置簇ID，层次，
    并对节点和边设置id(uuid)和human_readable_id(int)属性
    :param graphml: networkx graphml图
    :param communities: 节点分层分族结果
    :param level: 设置节点的层次
    :param seed: 随机种子
    :return: 更新后的图
    """
    random = Random(seed)
    # 加载原图
    graph = nx.parse_graphml(graphml)
    for community_level, community_id, nodes in communities:
        # 更新指定水平的节点的分层分簇
        if level == community_level:
            for node in nodes:
                graph.nodes[node]["cluster"] = community_id
                graph.nodes[node]["level"] = level
    # node_degree: (节点ID, 节点的度)
    for node_degree in graph.degree:
        # 设置节点的度属性
        graph.nodes[str(node_degree[0])]["degree"] = int(node_degree[1])
    # 新增节点ID属性
    for index, node in enumerate(graph.nodes()):
        graph.nodes[node]["human_readable_id"] = index
        graph.nodes[node]["id"] = str(gen_uuid(random))
    # 新增边ID属性
    for index, edge in enumerate(graph.edges()):
        graph.edges[edge]["id"] = str(gen_uuid(random))
        graph.edges[edge]["human_readable_id"] = index
        graph.edges[edge]["level"] = level

    return graph


def run_layout(
        strategy: dict[str, Any],
        graphml_or_graph: str | nx.Graph
) -> Communities:
    """
    对图的节点分层聚类
    :param strategy: 聚类策略
    :param graphml_or_graph: networkx图
    :return: 分层聚类结果：[(层次, 聚族ID, [节点ID])]
    """
    # 加载图
    graph = load_graph(graphml_or_graph)
    # 没有节点，无法聚类
    if len(graph.nodes) == 0:
        log.warning("Graph has no nodes.")
        return []

    clusters: dict[int, dict[str, list[str]]] = {}
    # 选择聚类算法
    strategy_type = strategy.get("type", GraphCommunityStrategyType.leiden)
    match strategy_type:
        case GraphCommunityStrategyType.leiden:
            from assistant.memory.graphrag_v1.index.verbs.graph.clustering.strategies.leiden import run as run_leiden
            # leiden分层聚类
            clusters = run_leiden(graph, strategy)
        case _:
            msg = f"Unknown clustering strategy {strategy_type}."
            raise ValueError(msg)

    # results: [(层次, 聚族ID, [节点ID])]
    results: Communities = []
    for level in clusters:
        for cluster_id, nodes in clusters[level].items():
            results.append((level, cluster_id, nodes))
    return results
