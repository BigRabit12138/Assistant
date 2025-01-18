import logging

from typing import cast

import pandas as pd

import assistant.memory.graphrag_v1.index.graph.extractors.community_reports.schemas as schemas

from assistant.memory.graphrag_v1.index.graph.extractors.community_reports.utils import set_context_size
from assistant.memory.graphrag_v1.index.graph.extractors.community_reports.sort_context import sort_context
from assistant.memory.graphrag_v1.index.graph.extractors.community_reports.build_mixed_context import build_mixed_context
from assistant.memory.graphrag_v1.index.utils.dataframes import (
    join,
    union,
    select,
    antijoin,
    drop_columns,
    transform_series,
    where_column_equals,
)

log = logging.getLogger(__name__)


def prep_community_report_context(
        report_df: pd.DataFrame | None,
        community_hierarchy_df: pd.DataFrame,
        local_context_df: pd.DataFrame,
        level: int | str,
        max_tokens: int,
) -> pd.DataFrame:
    """
    获取level层以及子层的聚簇的详细信息文本信息，community, all_context, context_string, level,
    context_size, context_exceed_limit
    :param report_df: 以及处理的聚簇
    :param community_hierarchy_df: 层间关系
    :param local_context_df: 所有的context
    :param level: 当前处理的层
    :param max_tokens: 最大token数量
    :return:
    """
    if report_df is None:
        report_df = pd.DataFrame()

    level = int(level)
    # 获取当前层的所有对象的详细信息
    level_context_df = _at_level(level, local_context_df)
    # 文本token没有超过限制的行
    valid_context_df = _within_context(level_context_df)
    # 文本token超过限制的行
    invalid_context_df = _exceeding_context(level_context_df)

    # 如果没有超长内容，直接返回
    if invalid_context_df.empty:
        return valid_context_df

    if report_df.empty:
        # 将超长的cntext_string截断
        invalid_context_df[schemas.CONTEXT_STRING] = _sort_and_trim_context(
            invalid_context_df, max_tokens
        )
        # 重新计算context_string的token数量
        set_context_size(invalid_context_df)
        # 修改超长标志
        invalid_context_df[schemas.CONTEXT_EXCEED_FLAG] = 0
        # 拼接，返回
        return union(valid_context_df, invalid_context_df)

    # 去掉已经被处理的community
    level_context_df = _antijoin_reports(level_context_df, report_df)

    # 获取下一个层的context
    sub_context_df = _get_subcontext_df(level + 1, report_df, local_context_df)
    # 获取下一层的详细信息
    community_df = _get_community_df(
        level,
        invalid_context_df,
        sub_context_df,
        community_hierarchy_df,
        max_tokens
    )
    # 超长的聚簇中排除下一层中出现的信息
    remaining_df = _antijoin_reports(invalid_context_df, community_df)
    # 更新还需要处理聚簇的contex_string
    remaining_df[schemas.CONTEXT_STRING] = _sort_and_trim_context(
        remaining_df, max_tokens
    )
    # 拼接所有聚簇
    result = union(valid_context_df, community_df, remaining_df)
    # 更新context_size
    set_context_size(result)
    # 设置不超长
    result[schemas.CONTEXT_EXCEED_FLAG] = 0
    return result


def _drop_community_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    从df中删掉level列
    :param df: 表格
    :return: 删掉level列的表格
    """
    return drop_columns(df, schemas.COMMUNITY_LEVEL)


def _at_level(level: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    获取指定level的行
    :param level: 指定的层次
    :param df: 表格
    :return: 保留了指定层次的表格
    """
    return where_column_equals(df, schemas.COMMUNITY_LEVEL, level)


def _exceeding_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    获取文本超过token数量限制的行
    :param df: 表格
    :return: 文本超过token数量限制的行的表格
    """
    return where_column_equals(df, schemas.CONTEXT_EXCEED_FLAG, 1)


def _within_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    保留文本token没有超过限制的行
    :param df: 表格
    :return: 保留了文本token没有超过限制的行的表格
    """
    return where_column_equals(df, schemas.CONTEXT_EXCEED_FLAG, 0)


def _antijoin_reports(
        df: pd.DataFrame,
        reports: pd.DataFrame
) -> pd.DataFrame:
    """
    从df中排除以及被reports收集的行
    :param df: 表格
    :param reports: 已经被收集的行
    :return: 排除以及被reports收集的行的表格
    """
    return antijoin(df, reports, schemas.NODE_COMMUNITY)


def _sort_and_trim_context(
        df: pd.DataFrame,
        max_tokens: int
) -> pd.Series:
    """
    对表格的all_context按边排序，以指定最大文本token的方式获取所有对象的详细文本信息
    :param df: 表格
    :param max_tokens: 最大token数量
    :return: all_context的文本的序列
    """
    series = cast(pd.Series, df[schemas.ALL_CONTEXT])
    return transform_series(series, lambda x: sort_context(x, max_tokens=max_tokens))


def _build_mixed_context(
        df: pd.DataFrame,
        max_tokens: int
) -> pd.Series:
    """
    获取子聚簇的文本信息
    :param df: 表格
    :param max_tokens: 最大token数量
    :return: 子聚簇的文本信息序列
    """
    series = cast(pd.Series, df[schemas.ALL_CONTEXT])
    return transform_series(
        series, lambda x: build_mixed_context(x, max_tokens=max_tokens)
    )


def _get_subcontext_df(
        level: int,
        report_df: pd.DataFrame,
        local_context_df: pd.DataFrame
) -> pd.DataFrame:
    """
    获取level的内容
    :param level: 层次
    :param report_df: 以及处理的context表格
    :param local_context_df: 当前处理的context的表格
    :return: 合并的表格
    """
    # 获取report的level层的内容，并删掉level列
    sub_report_df = _drop_community_level(_at_level(level, report_df))
    # 获取local_context_df的level层的内容
    sub_context_df = _at_level(level, local_context_df)
    # 在community上合并当前处理的contex和以及处理的context
    sub_context_df = join(sub_context_df, sub_report_df, schemas.NODE_COMMUNITY)
    sub_context_df.rename(
        columns={schemas.NODE_COMMUNITY: schemas.SUB_COMMUNITY}, inplace=True
    )
    return sub_context_df


def _get_community_df(
        level: int,
        invalid_context_df: pd.DataFrame,
        sub_context_df: pd.DataFrame,
        community_hierarchy_df: pd.DataFrame,
        max_tokens: int,
) -> pd.DataFrame:
    """
    获取level层子聚簇的文本信息，community, all_context, context_string, level
    :param level: 层次
    :param invalid_context_df: 超长的聚簇
    :param sub_context_df: 子聚簇
    :param community_hierarchy_df: 层间聚簇关系
    :param max_tokens: 最大token
    :return: level层子聚簇的文本信息
    """
    # 获取level层的下一层的聚簇信息，community, sub_community, sub_community_size
    community_df = _drop_community_level(_at_level(level, community_hierarchy_df))
    # context超长的行的community，community
    invalid_community_ids = select(invalid_context_df, schemas.NODE_COMMUNITY)
    # 获取下一层的sub_community, full_content, all_context, context_size
    subcontext_selection = select(
        sub_context_df,
        schemas.SUB_COMMUNITY,
        schemas.FULL_CONTENT,
        schemas.ALL_CONTEXT,
        schemas.CONTEXT_SIZE,
    )
    # 找到下一层中超长的community
    invalid_communities = join(
        community_df,
        invalid_community_ids,
        schemas.NODE_COMMUNITY,
        "inner"
    )
    # community, sub_community, sub_community_size, full_content, all_context, context_size
    community_df = join(
        invalid_communities,
        subcontext_selection,
        schemas.SUB_COMMUNITY
    )
    # 获取sub_community的详细信息
    community_df[schemas.ALL_CONTEXT] = community_df.apply(
        lambda x: {
            schemas.SUB_COMMUNITY: x[schemas.SUB_COMMUNITY],
            schemas.ALL_CONTEXT: x[schemas.ALL_CONTEXT],
            schemas.FULL_CONTENT: x[schemas.FULL_CONTENT],
            schemas.CONTEXT_SIZE: x[schemas.CONTEXT_SIZE],
        },
        axis=1,
    )
    # 按community分组，列表聚和all_context
    community_df = (
        community_df.groupby(schemas.NODE_COMMUNITY)
        .agg({schemas.ALL_CONTEXT: list})
        .reset_index()
    )
    # context_string更新子聚簇的文本信息
    community_df[schemas.CONTEXT_STRING] = _build_mixed_context(
        community_df, max_tokens
    )
    # 添加level
    community_df[schemas.COMMUNITY_LEVEL] = level
    return community_df
