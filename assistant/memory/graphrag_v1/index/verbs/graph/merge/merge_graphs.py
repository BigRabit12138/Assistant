from typing import Any, cast

import networkx as nx
import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    VerbCallbacks,
    TableContainer,
    progress_iterable,
)

from assistant.memory.graphrag_v1.index.utils import load_graph
from assistant.memory.graphrag_v1.index.verbs.graph.merge.defaults import (
    DEFAULT_NODE_OPERATIONS,
    DEFAULT_EDGE_OPERATIONS,
    DEFAULT_CONCAT_SEPARATOR,
)
from assistant.memory.graphrag_v1.index.verbs.graph.merge.typing import (
    StringOperation,
    NumericOperation,
    BasicMergeOperation,
    DetailedAttributeMergeOperation,
)


@verb(name="merge_graphs")
def merge_graphs(
        input: VerbInput,
        callbacks: VerbCallbacks,
        column: str,
        to: str,
        nodes: dict[str, Any] = DEFAULT_NODE_OPERATIONS,
        edges: dict[str, Any] = DEFAULT_EDGE_OPERATIONS,
        **_kwargs,
) -> TableContainer:
    """
    将column列的图合并到一个大图，并赋值给新的表格的to列
    :param input: 输入，包含输入表格
    :param callbacks: 回调钩子
    :param column: 源图所在列
    :param to: 最终大图所在列
    :param nodes: 节点合并操作
    :param edges: 边合并操作
    :param _kwargs: 额外操作
    :return: 输出，包含输出表格
    """
    input_df = input.get_input()
    output = pd.DataFrame()

    # 获取节点的属性合并操作
    node_ops = {
        attrib: _get_detailed_attribute_merge_operation(value)
        for attrib, value in nodes.items()
    }
    # 获取边的属性合并操作
    edge_ops = {
        attrib: _get_detailed_attribute_merge_operation(value)
        for attrib, value in edges.items()
    }
    mega_graph = nx.Graph()
    num_total = len(input_df)
    for graphml in progress_iterable(input_df[column], callbacks.progress, num_total):
        # 加载图
        graph = load_graph(cast(str | nx.Graph, graphml))
        # 合并节点
        merge_nodes(mega_graph, graph, node_ops)
        # 合并边
        merge_edges(mega_graph, graph, edge_ops)

    output[to] = ["\n".join(nx.generate_graphml(mega_graph))]

    return TableContainer(table=output)


def merge_nodes(
        target: nx.Graph,
        subgraph: nx.Graph,
        node_ops: dict[str, DetailedAttributeMergeOperation],
):
    """
    将subgraph的节点合并到target
    :param target: 目标图
    :param subgraph: 子图
    :param node_ops: 合并操作
    :return:
    """
    for node in subgraph.nodes:
        if node not in target.nodes:
            # 新节点，创建
            target.add_node(node, **(subgraph.nodes[node] or {}))
        else:
            # 已经存在，合并
            merge_attributes(target.nodes[node], subgraph.nodes[node], node_ops)


def merge_edges(
        target_graph: nx.Graph,
        subgraph: nx.Graph,
        edge_ops: dict[str, DetailedAttributeMergeOperation],
):
    """
    将subgraph的边合并到target_graph
    :param target_graph: 目标图
    :param subgraph: 子图
    :param edge_ops: 合并操作
    :return:
    """
    for source, target, edge_data in subgraph.edges(data=True):
        if not target_graph.has_edge(source, target):
            # 新边，创建
            target_graph.add_edge(source, target, **(edge_data or {}))
        else:
            # 已经存在，合并
            merge_attributes(target_graph.edges[(source, target)], edge_data, edge_ops)


def merge_attributes(
        target_item: dict[str, Any] | None,
        source_item: dict[str, Any] | None,
        ops: dict[str, DetailedAttributeMergeOperation],
):
    """
    将源节点的属性合并到目标节点
    :param target_item: 目标节点
    :param source_item: 源节点
    :param ops: 合并操作
    :return:
    """
    source_item = source_item or {}
    target_item = target_item or {}
    for op_attrib, op in ops.items():
        # 匹配任意属性
        if op_attrib == "*":
            for attrib in source_item:
                # 排除以及有具体合并方法的属性
                if attrib not in ops:
                    # 对一个属性进行合并
                    apply_merge_operation(target_item, source_item, attrib, op)
        else:
            if op_attrib in source_item or op_attrib in target_item:
                apply_merge_operation(target_item, source_item, op_attrib, op)


def apply_merge_operation(
        target_item: dict[str, Any] | None,
        source_item: dict[str, Any] | None,
        attrib: str,
        op: DetailedAttributeMergeOperation,
):
    """
    将source_item的attrib合并到target_item
    :param target_item: 目标节点
    :param source_item: 源节点
    :param attrib: 合并属性
    :param op: 合并操作
    :return:
    """
    source_item = source_item or {}
    target_item = target_item or {}

    # 替换
    if (
        op.operation == BasicMergeOperation.Replace
        or op.operation == StringOperation.Replace
    ):
        target_item[attrib] = source_item.get(attrib, None) or ""
    # 忽略
    elif (
        op.operation == BasicMergeOperation.Skip
        or op.operation == StringOperation.Skip
    ):
        target_item[attrib] = target_item.get(attrib, None) or ""
    # 拼接
    elif op.operation == StringOperation.Concat:
        separator = op.separator or DEFAULT_CONCAT_SEPARATOR
        target_attrib = target_item.get(attrib, "") or ""
        source_attrib = source_item.get(attrib, "") or ""
        target_item[attrib] = f"{target_attrib}{separator}{source_attrib}"
        # 去重
        if op.distinct:
            target_item[attrib] = separator.join(
                sorted(set(target_item[attrib].split(separator)))
            )
    # 加
    elif op.operation == NumericOperation.Sum:
        target_item[attrib] = (target_item.get(attrib, 0) or 0) + (
            source_item.get(attrib, 0) or 0
        )
    # 取平均
    elif op.operation == NumericOperation.Average:
        target_item[attrib] = (
            (target_item.get(attrib, 0) or 0) or (source_item.get(attrib, 0) or 0)
        ) / 2
    # 取最大值
    elif op.operation == NumericOperation.Max:
        target_item[attrib] = max(
            (target_item.get(attrib, 0) or 0), (source_item.get(attrib, 0) or 0)
        )
    # 取最小值
    elif op.operation == NumericOperation.Min:
        target_item[attrib] = min(
            (target_item.get(attrib, 0) or 0), (source_item.get(attrib, 0) or 0)
        )
    # 取乘积
    elif op.operation == NumericOperation.Multiply:
        target_item[attrib] = (target_item.get(attrib, 1) or 1) * (
            source_item.get(attrib, 1) or 1
        )
    else:
        msg = f"Invalid operation {op.operation}."
        raise ValueError(msg)


def _get_detailed_attribute_merge_operation(
        value: str | dict[str, Any],
) -> DetailedAttributeMergeOperation:
    """
    解析属性合并操作
    :param value: 操作类型
    :return: 属性合并操作
    """
    if isinstance(value, str):
        return DetailedAttributeMergeOperation(operation=value)
    return DetailedAttributeMergeOperation(**value)
