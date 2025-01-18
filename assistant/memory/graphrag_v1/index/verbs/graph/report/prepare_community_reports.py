import logging

from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    VerbCallbacks,
    TableContainer,
    progress_iterable,
)

import assistant.memory.graphrag_v1.index.graph.extractors.community_reports.schemas as schemas

from assistant.memory.graphrag_v1.index.utils.ds_util import (
    get_named_input_table,
    get_required_input_table,
)
from assistant.memory.graphrag_v1.index.graph.extractors.community_reports import (
    get_levels,
    sort_context,
    set_context_size,
    filter_nodes_to_level,
    filter_edges_to_nodes,
    filter_claims_to_nodes,
    set_context_exceeds_flag,
)

log = logging.getLogger(__name__)


@verb(name="prepare_community_reports")
def prepare_community_reports(
        input: VerbInput,
        callbacks: VerbCallbacks,
        max_tokens: int = 16_000,
        **_kwargs,
) -> TableContainer:
    """
    获取所有层的所有聚簇的对象的详细文本信息
    :param input: 输入，包含表格
    :param callbacks: 回调钩子
    :param max_tokens: 最大token限制
    :param _kwargs: 额外参数
    :return: 输出表格，community, all_context, context_string, context_size,
    context_exceed_limit, level
    """
    node_df = cast(pd.DataFrame, get_required_input_table(input, "nodes").table)
    edge_df = cast(pd.DataFrame, get_required_input_table(input, "edges").table)
    claim_df = get_named_input_table(input, "claims")
    if claim_df is not None:
        claim_df = cast(pd.DataFrame, claim_df.table)

    # 获取所有的level
    levels = get_levels(node_df, schemas.NODE_LEVEL)
    dfs = []

    for level in progress_iterable(levels, callbacks.progress, len(levels)):
        # 获取当前层的所有聚簇的对象的详细文本信息
        communities_at_level_df = _prepare_reports_at_level(
            node_df, edge_df, claim_df, level, max_tokens
        )
        dfs.append(communities_at_level_df)

    return TableContainer(table=pd.concat(dfs))


def _prepare_reports_at_level(
        node_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        claim_df: pd.DataFrame | None,
        level: int,
        max_tokens: int = 16_000,
        community_id_column: str = schemas.COMMUNITY_ID,
        node_id_column: str = schemas.NODE_ID,
        node_name_column: str = schemas.NODE_NAME,
        node_details_column: str = schemas.NODE_DETAILS,
        node_level_column: str = schemas.NODE_LEVEL,
        node_degree_column: str = schemas.NODE_DEGREE,
        node_community_column: str = schemas.NODE_COMMUNITY,
        edge_id_column: str = schemas.EDGE_ID,
        edge_source_column: str = schemas.EDGE_SOURCE,
        edge_target_column: str = schemas.EDGE_TARGET,
        edge_degree_column: str = schemas.EDGE_DEGREE,
        edge_details_column: str = schemas.EDGE_DETAILS,
        claim_id_column: str = schemas.CLAIM_ID,
        claim_subject_column: str = schemas.CLAIM_SUBJECT,
        claim_details_column: str = schemas.CLAIM_DETAILS,
):
    """
    获取当前level的所有community的对象的详细文本信息
    :param node_df: 所有节点表格
    :param edge_df: 所有边表格
    :param claim_df: 所有claim表格
    :param level: 当前层次
    :param max_tokens: 文本最大token限制
    :param community_id_column: 聚簇id
    :param node_id_column: 节点human_readable_id所在列
    :param node_name_column: 节点名字所在列
    :param node_details_column: 节点详细信息所在列
    :param node_level_column: 节点所属层次所在列
    :param node_degree_column: 节点的度所在列
    :param node_community_column: 节点的聚簇所在列
    :param edge_id_column: 边的human_readable_id所在列
    :param edge_source_column: 边的源点所在列
    :param edge_target_column: 边的目的点所在列
    :param edge_degree_column: 边的度所在列
    :param edge_details_column: 边的详细信息所在列
    :param claim_id_column: claim human_readable_id所在列
    :param claim_subject_column: claim所属对象节点所在列
    :param claim_details_column: claim详细信息所在列
    :return: 输出表格，community, all_context, context_string, context_size,
    context_exceed_limit, level
    """
    def get_edge_details(
            node_df_: pd.DataFrame,
            edge_df_: pd.DataFrame,
            name_col: str
    ):
        """
        获取边的详细信息，并和节点在节点名字上拼接
        :param node_df_: 节点表格
        :param edge_df_: 边表格
        :param name_col: 边的源点或者目标点
        :return: 附件了边信息的节点表格
        """
        return node_df_.merge(
            cast(
                pd.DataFrame,
                edge_df_[[name_col, schemas.EDGE_DETAILS]],
            ).rename(columns={name_col: schemas.NODE_NAME}),
            on=schemas.NODE_NAME,
            how="left",
        )
    # 获取指定层次的节点
    level_node_df = filter_nodes_to_level(node_df, level)
    log.info(f"Number of nodes at level={level} => {len(level_node_df)}.")
    # 获取所有节点的名字
    nodes = level_node_df[node_name_column].tolist()
    # 获取出现在节点里面的边
    level_edge_df = filter_edges_to_nodes(edge_df, nodes)
    level_claim_df = (
        filter_claims_to_nodes(claim_df, nodes)
        if claim_df is not None else None
    )
    # 拼接所有边的信息
    merged_node_df = pd.concat(
        [
            get_edge_details(level_node_df, level_edge_df, edge_source_column),
            get_edge_details(level_node_df, level_edge_df, edge_target_column),
        ],
        axis=0,
    )
    # 按title, community, degree, level分组，
    # node_details保留第一个，edge_details转换为列表，并重置索引
    merged_node_df = (
        merged_node_df.groupby(
            [
                node_name_column,
                node_community_column,
                node_degree_column,
                node_level_column,
            ]
        )
        .agg({node_details_column: "first", edge_details_column: list})
        .reset_index()
    )

    if level_claim_df is not None:
        merged_node_df = merged_node_df.merge(
            cast(
                pd.DataFrame,
                level_claim_df[[claim_subject_column, claim_details_column]],
            ).rename(columns={claim_subject_column: node_name_column}),
            on=node_name_column,
            how="left",
        )

    # 按title, community, level, degree分组，
    # node_details保留第一个，edge_details保留第一个，
    # claim_details转换为列表，并重置索引
    merged_node_df = (
        merged_node_df.groupby(
            [
                node_name_column,
                node_community_column,
                node_level_column,
                node_degree_column,
            ]
        )
        .agg(
            {
                node_details_column: "first",
                edge_details_column: "first",
                **({claim_details_column: list} if level_claim_df is not None else {}),
            }
        )
        .reset_index()
    )
    # 获取节点的title, degree, node_details, edge_details, claim_details，造成字典
    # 赋给all_context列
    merged_node_df[schemas.ALL_CONTEXT] = merged_node_df.apply(
        lambda x: {
            node_name_column: x[node_name_column],
            node_degree_column: x[node_degree_column],
            node_details_column: x[node_details_column],
            edge_details_column: x[edge_details_column],
            claim_details_column: x[claim_details_column]
            if level_claim_df is not None else [],
        },
        axis=1
    )
    # 按community分组，将all_context列聚合为列表，并重置索引
    community_df = (
        merged_node_df.groupby(node_community_column)
        .agg({schemas.ALL_CONTEXT: list})
        .reset_index()
    )
    # 获取一个聚簇内所有对象的details的文本，并赋给context_string列
    community_df[schemas.CONTEXT_STRING] = community_df[schemas.ALL_CONTEXT].apply(
        lambda x: sort_context(
            x,
            node_id_column=node_id_column,
            node_name_column=node_name_column,
            node_details_column=node_details_column,
            edge_id_column=edge_id_column,
            edge_details_column=edge_details_column,
            edge_degree_column=edge_degree_column,
            edge_source_column=edge_source_column,
            edge_target_column=edge_target_column,
            claim_id_column=claim_id_column,
            claim_details_column=claim_details_column,
            community_id_column=community_id_column,
        )
    )
    # 计算表格的context_string列的token，并赋值给context_size列
    set_context_size(community_df)
    # 添加是否超长的标志的表格
    set_context_exceeds_flag(community_df, max_tokens)
    # 添加聚簇所在的level的列
    community_df[schemas.COMMUNITY_LEVEL] = level
    return community_df
