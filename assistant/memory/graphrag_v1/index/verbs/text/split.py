from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)


@verb(name="text_split")
def text_split(
        input: VerbInput,
        column: str,
        to: str,
        separator: str = ",",
        **_kwargs: dict,
) -> TableContainer:
    """
    将column列的内容分割成列表赋值给to列
    :param input: 输入，包含操作的表
    :param column: 分割的列
    :param to: 新列
    :param separator: 分割符号
    :return: 输出，包含表格
    """
    output = text_split_df(
        cast(pd.DataFrame, input.get_input()),
        column,
        to,
        separator,
    )
    return TableContainer(table=output)


def text_split_df(
        input_: pd.DataFrame,
        column: str,
        to: str,
        separator: str = ","
) -> pd.DataFrame:
    """
    将column列的内容分割成列表赋值给to列
    :param input_: 操作的表
    :param column: 分割的列
    :param to: 新列
    :param separator: 分割符号
    :return: 操作的表
    """
    output = input_

    def _apply_split(row):
        """
        对column列分割成列表
        :param row: 每一行
        :return: 列表
        """
        # 空或以及是独立元素，直接返回
        if row[column] is None or isinstance(row[column], list):
            return row[column]
        if row[column] == "":
            return []
        if not isinstance(row[column], str):
            message = f"Expected {column} to be a string, but got {type(row[column])}."
            raise TypeError(message)

        return row[column].split(separator)
    # 赋值给to列
    output[to] = output.apply(_apply_split, axis=1)
    return output
