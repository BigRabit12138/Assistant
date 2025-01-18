import html

import networkx as nx


def normalize_node_names(graph: nx.Graph | nx.DiGraph) -> nx.Graph | nx.DiGraph:
    """
    将图的节点的名字转换为UTF字符集
    :param graph: networkx图
    :return: 换名后的图
    """
    # 将html的字符翻译成utf字符
    node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes}
    # 将图的节点的key用转换后的字符命名
    return nx.relabel_nodes(graph, node_mapping)
