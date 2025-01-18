import networkx as nx


def load_graph(graphml: str | nx.Graph) -> nx.Graph:
    """
    加载一个networkx图对象
    :param graphml: graphml图
    :return: Graph图对象
    """
    return nx.parse_graphml(graphml) \
        if isinstance(graphml, str) else graphml
