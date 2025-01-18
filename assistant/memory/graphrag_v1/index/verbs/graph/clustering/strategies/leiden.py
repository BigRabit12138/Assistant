import logging

from typing import Any

import networkx as nx

from graspologic.partition import hierarchical_leiden

from assistant.memory.graphrag_v1.index.graph.utils import stable_largest_connected_component

log = logging.getLogger(__name__)


def run(
        graph: nx.Graph,
        args: dict[str, Any]
) -> dict[int, dict[str, list[str]]]:
    """
    使用leiden算法对图的节点进行分层聚簇
    :param graph: networkx图
    :param args: leiden算法参数
    :return: 分层聚簇结果：{层次: {节点聚簇ID: [节点ID]}}
    """
    max_cluster_size = args.get("max_cluster_size", 10)
    use_lcc = args.get("use_lcc", True)
    if args.get("verbose", False):
        log.info(
            f"Running leiden with max_cluster_size={max_cluster_size}, lcc={use_lcc}."
        )
    # 使用leiden算方法分层聚类
    node_id_to_community_map = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=args.get("seed", 0xDEADBEEF),
    )
    levels = args.get("levels")

    # 排序划分层次
    if levels is None:
        levels = sorted(node_id_to_community_map.keys())
    # results_by_level: {层次: {节点聚簇ID: [节点ID]}}
    results_by_level: dict[int, dict[str, list[str]]] = {}
    # 对聚簇进行分类
    for level in levels:
        # 初始化层
        result = {}
        results_by_level[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = str(raw_community_id)
            # 初始化簇
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)
    return results_by_level


def _compute_leiden_communities(
        graph: nx.Graph | nx.DiGraph,
        max_cluster_size: int,
        use_lcc: bool,
        seed=0xDEADBEEF,
) -> dict[int, dict[str, int]]:
    """
    使用leiden算方法分层聚类
    :param graph: networkx图
    :param max_cluster_size: 最大聚类数量
    :param use_lcc: 是否压缩图为连通子图
    :param seed: 随机种子
    :return: 层次划分即聚类结果：{层次: {节点ID: 节点所属的族的ID}}
    """
    # 获取连通子图，转换html字符，并排序
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    # 使用leiden算法进行层次化划分
    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    # results: {层次: {节点ID: 节点所属的族的ID}}
    results: dict[int, dict[str, int]] = {}
    for partition in community_mapping:
        # 如何未添加，初始化
        results[partition.level] = results.get(partition.level, {})
        # 将本次划分的节点添加进同一水平
        results[partition.level][partition.node] = partition.cluster

    return results
