from typing import Any, cast
from dataclasses import dataclass

import pandas as pd

from datashaper import (
    verb,
    Progress,
    VerbInput,
    VerbCallbacks,
    TableContainer,
    FieldAggregateOperation,
    aggregate_operation_mapping,
)

ARRAY_AGGREGATIONS = [
    FieldAggregateOperation.ArrayAgg,
    FieldAggregateOperation.ArrayAggDistinct,
]


@verb(name="aggregate_override")
def aggregate(
        input: VerbInput,
        callbacks: VerbCallbacks,
        aggregations: list[dict[str, Any]],
        groupby: list[str] | None = None,
        **_kwargs: dict,
) -> TableContainer:
    """
    对pandas表格在groupby列分组，并进行聚合，并使用新的名字替换，
    只保留聚合的列和索引的列
    :param input: 输入，包含输入表格
    :param callbacks: 回调器
    :param aggregations: 聚合操作
    :param groupby: 分组列
    :param _kwargs: 额外参数
    :return: 输出，包含输入表格
    """
    # 解析聚合策略
    aggregations_to_apply = _load_aggregations(aggregations)
    df_aggregations = {
        agg.column: _get_pandas_agg_operation(agg)
        for agg in aggregations_to_apply.values()
    }

    input_table = input.get_input()
    callbacks.progress(Progress(percent=0))

    # 分组
    if groupby is None:
        output_grouped = input_table.groupby(lambda _x: True)
    else:
        output_grouped = input_table.groupby(groupby, sort=False)
    # 聚合
    output = cast(pd.DataFrame, output_grouped.agg(df_aggregations))
    # 覆盖
    output.rename(
        columns={agg.column: agg.to for agg in aggregations_to_apply.values()},
        inplace=True,
    )
    output.columns = [agg.to for agg in aggregations_to_apply.values()]

    callbacks.progress(Progress(percent=1))

    return TableContainer(table=output.reset_index())


@dataclass
class Aggregation:
    """
    聚合操作数据类
    """
    # 操作列
    column: str | None
    # 操作类型
    operation: str
    # 目标列
    to: str

    separator: str | None = None


def _get_pandas_agg_operation(agg: Aggregation) -> Any:
    """
    获取pandas表格聚合方法
    :param agg: 聚合策略
    :return: 聚合可调用函数或者函数名
    """
    if agg.operation == "string_concat":
        return (agg.separator or ',').join

    return aggregate_operation_mapping[FieldAggregateOperation(agg.operation)]


def _load_aggregations(
        aggregations: list[dict[str, Any]],
) -> dict[str, Aggregation]:
    """
    解析聚合参数
    :param aggregations: 聚合参数
    :return: 解析后的聚合参数
    """
    return {
        aggregation["column"]: Aggregation(
            aggregation["column"], aggregation["operation"], aggregation["to"]
        )
        for aggregation in aggregations
    }
