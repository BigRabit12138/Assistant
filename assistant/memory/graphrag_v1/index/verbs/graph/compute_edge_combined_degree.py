from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)

from assistant.memory.graphrag_v1.index.utils.ds_util import get_required_input_table


@verb(name="compute_edge_combined_degree")
def compute_edge_combined_degree(
        input: VerbInput,
        to: str = "rank",
        node_name_column: str = "title",
        node_degree_column: str = "degree",
        edge_source_column: str = "source",
        edge_target_column: str = "target",
        **_kwargs,
) -> TableContainer:
    """
    计算边的度
    :param input: 输入包含表格
    :param to: 保存度的列
    :param node_name_column: 节点名字的列
    :param node_degree_column: 节点的度的列
    :param edge_source_column: 源点的列
    :param edge_target_column: 目的点的列
    :param _kwargs: 额外参数
    :return: 添加了度的表
    """
    edge_df: pd.DataFrame = cast(pd.DataFrame, input.get_input())
    if to in edge_df.columns:
        return TableContainer(table=edge_df)
    # 获取节点的度
    node_degree_df = _get_node_degree_table(
        input, node_name_column, node_degree_column
    )

    def join_to_degree(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        将node_degree_df节点的度合并到df上
        :param df: 待合并的表
        :param column: df要合并的列
        :return: 添加了度的df
        """
        # 保存度的列
        degree_column = _degree_colname(column)
        # 将节点的node_name_column改名为column，node_degree_column改名为degree_column
        # 将df和改名后的节点在column上合并，保留df
        # 相当与给df添加保存度的列
        result = df.merge(
            node_degree_df.rename(
                columns={node_name_column: column, node_degree_column: degree_column}
            ),
            on=column,
            how="left",
        )
        # 将NA赋值0
        result[degree_column] = result[degree_column].fillna(0)
        return result
    # 给edge_source_column的边添加度
    edge_df = join_to_degree(edge_df, edge_source_column)
    # 给edge_target_column的边添加度
    edge_df = join_to_degree(edge_df, edge_target_column)
    # 获取总的度，赋值给to列
    edge_df[to] = (
        edge_df[_degree_colname(edge_source_column)]
        + edge_df[_degree_colname(edge_target_column)]
    )

    return TableContainer(table=edge_df)


def _degree_colname(column: str) -> str:
    """
    添加_degree后缀
    :param column: 列名
    :return: 带_degree后缀的列名
    """
    return f"{column}_degree"


def _get_node_degree_table(
        input_: VerbInput,
        node_name_column: str,
        node_degree_column: str
) -> pd.DataFrame:
    """
    获取名为nodes的表，并保留node_name_column, node_degree_column两列
    :param input_: 输入，包含表格
    :param node_name_column: 节点名字所在列
    :param node_degree_column: 节点度所在列
    :return: 包含名字、度的所有节点
    """
    # 获取filtered_nodes动作的表
    nodes_container = get_required_input_table(input_, "nodes")
    nodes = cast(pd.DataFrame, nodes_container.table)
    # 保留node_name_column, node_degree_column两列
    return cast(pd.DataFrame, nodes[[node_name_column, node_degree_column]])
