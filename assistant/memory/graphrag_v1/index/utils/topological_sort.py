from graphlib import TopologicalSorter


def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    ts = TopologicalSorter(graph)
    return list(ts.static_order())
