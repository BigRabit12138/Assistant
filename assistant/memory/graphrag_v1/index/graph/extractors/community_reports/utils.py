from typing import cast

import pandas as pd

import assistant.memory.graphrag_v1.index.graph.extractors.community_reports.schemas as schemas

from assistant.memory.graphrag_v1.query.llm.text_utils import num_tokens


def set_context_size(df: pd.DataFrame) -> None:
    """
    计算表格的context_string列的token，并赋值给context_size列
    :param df: 表格
    :return: 添加了context_size列的表格
    """
    df[schemas.CONTEXT_SIZE] = df[schemas.CONTEXT_STRING].apply(lambda x: num_tokens(x))


def set_context_exceeds_flag(
        df: pd.DataFrame,
        max_tokens: int
) -> None:
    """
    判断是否超过token限制
    :param df: 表格
    :param max_tokens: 最大token
    :return: 添加是否超长的标志的表格
    """
    df[schemas.CONTEXT_EXCEED_FLAG] = df[schemas.CONTEXT_SIZE].apply(
        lambda x: x > max_tokens
    )


def get_levels(
        df: pd.DataFrame,
        level_column: str = schemas.NODE_LEVEL
) -> list[int]:
    """
    获取倒序排列的所有level
    :param df: 表格
    :param level_column: level所在列
    :return: 倒序排列的所有level
    """
    # 获取倒序排列的所有level，NA用-1填充
    result = sorted(df[level_column].fillna(-1).unique().tolist(), reverse=True)
    # 排除-1
    return [r for r in result if r != -1]


def filter_nodes_to_level(
        node_df: pd.DataFrame,
        level: int
) -> pd.DataFrame:
    """
    获取所有为level的节点
    :param node_df: 节点表格
    :param level: 节点层次
    :return: 所有为level的节点
    """
    return cast(pd.DataFrame, node_df[node_df[schemas.NODE_LEVEL] == level])


def filter_edges_to_nodes(
        edge_df: pd.DataFrame,
        nodes: list[str]
) -> pd.DataFrame:
    """
    获取出现在节点里面的边
    :param edge_df: 边表格
    :param nodes: 所有节点的名字
    :return: 出现在节点里面的边
    """
    return cast(
        pd.DataFrame,
        edge_df[
            edge_df[schemas.EDGE_SOURCE].isin(nodes)
            & edge_df[schemas.EDGE_TARGET].isin(nodes)
        ],
    )


def filter_claims_to_nodes(
        claims_df: pd.DataFrame,
        nodes: list[str]
) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        claims_df[claims_df[schemas.CLAIM_SUBJECT].isin(nodes)],
    )
