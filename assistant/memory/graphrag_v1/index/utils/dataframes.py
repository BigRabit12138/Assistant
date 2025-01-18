from typing import Any, cast
from collections.abc import Callable

import pandas as pd

from pandas._typing import MergeHow


def drop_columns(
        df: pd.DataFrame,
        *column: str
) -> pd.DataFrame:
    """
    删除指定的列
    :param df: 表格
    :param column: 列
    :return: 删除指定的列的表格
    """
    return df.drop(list(column), axis=1)


def where_column_equals(
        df: pd.DataFrame,
        column: str,
        value: Any
) -> pd.DataFrame:
    """
    保留df中column列值为value的行
    :param df: 表格
    :param column: 列名
    :param value: 值
    :return: 保留df中column列值为value的行的表格
    """
    return cast(pd.DataFrame, df[df[column] == value])


def antijoin(
        df: pd.DataFrame,
        exclude: pd.DataFrame,
        column: str
) -> pd.DataFrame:
    """
    从df中排除等于exclude的column列值的行
    :param df: 表格
    :param exclude: 待排除的表格
    :param column: 列名
    :return: df中排除等于exclude的column列值的行
    """
    result = df.merge(
        exclude[[column]],
        on=column,
        how="outer",
        indicator=True,
    )
    if "_merge" in result.columns:
        result = result[result["_merge"] == "left_only"].drop("_merge", axis=1)
    return cast(pd.DataFrame, result)


def transform_series(
        series: pd.Series,
        fn: Callable[[Any], Any]
) -> pd.Series:
    """
    对series的每个元素应用fn
    :param series: 序列对象
    :param fn: 处理函数
    :return: 处理后的序列对象
    """
    return cast(pd.Series, series.apply(fn))


def join(
        left: pd.DataFrame,
        right: pd.DataFrame,
        key: str,
        strategy: MergeHow = "left"
) -> pd.DataFrame:
    """
    将left和right表格以左连接的方式在key列上合并
    :param left: 表格
    :param right: 表格
    :param key: 列名
    :param strategy: 合并方式
    :return: 合并的表格
    """
    return left.merge(right, on=key, how=strategy)


def union(*frames: pd.DataFrame) -> pd.DataFrame:
    """
    拼接多个表格
    :param frames: 多个表格
    :return: 拼接的表格
    """
    return pd.concat(list(frames))


def select(df: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """
    选择指定列
    :param df: 表格
    :param columns: 列名
    :return: 选定列的表格
    """
    return cast(pd.DataFrame, df[list(columns)])
