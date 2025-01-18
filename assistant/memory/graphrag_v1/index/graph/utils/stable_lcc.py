from typing import Any, cast

import networkx as nx

from graspologic.utils import largest_connected_component

from assistant.memory.graphrag_v1.index.graph.utils.normalize_node_names import normalize_node_names


def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """
    找到图中的最大连通分量并对其节点和边排序
    :param graph: 图
    :return: 排序并转换html字符后的连同分量
    """
    graph = graph.copy()
    # 找到图中的最大连通分量
    graph = cast(nx.Graph, largest_connected_component(graph))
    # 将图的节点的名字换为UTF字符
    graph = normalize_node_names(graph)
    # 排序图的节点和边
    return _stabilize_graph(graph)


def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
    """
    对图的节点和边按ID排序
    :param graph: networkx图
    :return: 排序后的图
    """
    # 使用有向图或者无向图
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

    # data=True获取附带的数据
    sorted_nodes = graph.nodes(data=True)
    # 按节点ID排序
    sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])
    fixed_graph.add_nodes_from(sorted_nodes)

    edges = list(graph.edges(data=True))
    if not graph.is_directed():

        def _sort_source_target(edge):
            """
            边从小的指向大的
            :param edge: 边
            :return: 排序后的边
            """
            source, target, edge_data = edge
            if source > target:
                source, target = target, source
            return source, target, edge_data
        # 对每一条边的起点终点排序
        edges = [_sort_source_target(edge) for edge in edges]

    def _get_edge_key(source: Any, target: Any) -> str:
        """
        获取边的ID
        :param source: 源节点
        :param target: 目标节点
        :return: 边ID
        """
        return f"{source} -> {target}"
    # 按定义的边的ID对所有的边进行排序
    edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))
    fixed_graph.add_edges_from(edges)

    return fixed_graph
