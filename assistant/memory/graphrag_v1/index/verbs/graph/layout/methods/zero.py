import logging
import traceback

from typing import Any

import networkx as nx

from assistant.memory.graphrag_v1.index.typing import ErrorHandlerFn
from assistant.memory.graphrag_v1.index.graph.visualization import (
    GraphLayout,
    NodePosition,
    get_zero_positions,
)

log = logging.getLogger(__name__)


def run(
        graph: nx.Graph,
        _args: dict[str, Any],
        on_error: ErrorHandlerFn,
) -> GraphLayout:
    """
    获取图节点的空间布局，位置放置在2维原点
    :param graph: networkx图
    :param _args: 运行参数
    :param on_error: 错误回调
    :return: 图节点布局
    """
    node_clusters = []
    node_sizes = []

    nodes = list(graph.nodes)

    # 获取所有节点的簇和度数据
    for node_id in nodes:
        node = graph.nodes[node_id]
        cluster = node.get("cluster", node.get("community", -1))
        node_clusters.append(cluster)
        size = node.get("degree", node.get("size", 0))
        node_sizes.append(size)

    additional_args = {}
    if len(node_clusters) > 0:
        additional_args["node_categories"] = node_clusters
    if len(node_sizes) > 0:
        additional_args["node_sizes"] = node_sizes

    try:
        # 获取节点布局，放置在2维原点
        return get_zero_positions(node_labels=nodes, **additional_args)
    # TODO: 好像不会出错的
    except Exception as e:
        log.exception("Error running zero-position.")
        on_error(e, traceback.format_exc(), None)
        result = []
        for i in range(len(nodes)):
            cluster = node_clusters[i] if len(node_clusters) > 0 else 1
            result.append(
                NodePosition(
                    x=0,
                    y=0,
                    label=nodes[i],
                    size=0,
                    cluster=str(cluster),
                )
            )
        return result
