from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)


@verb(name="unzip")
def unzip(
        input: VerbInput,
        column: str,
        to: list[str],
        **_kwargs: dict
) -> TableContainer:
    """
    将输入表格的列中的内容解包成单列
    :param input: 输入，包含表格
    :param column: 解包的列
    :param to: 存放解包内容的列
    :param _kwargs: 额外的参数
    :return: 输出，包含表格
    """
    table = cast(pd.DataFrame, input.get_input())

    table[to] = pd.DataFrame(table[column].tolist(), index=table.index)

    return TableContainer(table=table)

