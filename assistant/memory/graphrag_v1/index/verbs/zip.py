from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)


@verb(name="zip")
def zip_verb(
        input: VerbInput,
        to: str,
        columns: list[str],
        type_: str | None = None,
        **_kwargs: dict,
) -> TableContainer:
    """
    将指定的列打包成元组或者字典
    :param input: 输入，包含输入表格
    :param to: 目标列名
    :param columns: 打包的列
    :param type_: 打包类型，默认元组，可以选择字典
    :param _kwargs: 额外参数
    :return: 输出，包含输出表格
    """
    table = cast(pd.DataFrame, input.get_input())
    if type_ is None:
        table[to] = list(zip(*[table[col] for col in columns], strict=True))

    elif type_ == "dict":
        if len(columns) != 2:
            msg = f"Expected exactly two columns for a dict, got {columns}."
            raise ValueError(msg)

        key_col, value_col = columns

        results = []
        for _, row in table.iterrows():
            keys = row[key_col]
            values = row[value_col]
            output = {}
            if len(keys) != len(values):
                msg = f"Expected same number of keys and values, got {len(keys)} keys and {len(values)} values."
                raise ValueError(msg)
            for idx, key in enumerate(keys):
                output[key] = values[idx]
            results.append(output)
        table[to] = results
    return TableContainer(table=table.reset_index(drop=True))
