from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)

from assistant.memory.graphrag_v1.index.utils import gen_md5_hash


@verb(name="genid")
def genid(
        input: VerbInput,
        to: str,
        method: str = "md5_hash",
        hash: list[str] = [],
        **_kwargs: dict,
) -> TableContainer:
    """
    生成ID到to列
    :param input: 输入，包含表格
    :param to: 存放ID的列
    :param method: 生成ID的方法
    :param hash: 如果ID是md5 hash，指定hash输入的列
    :param _kwargs: 额外的参数
    :return: 输出，包含表格
    """

    data = cast(pd.DataFrame, input.source.table)

    # hash ID
    if method == "md5_hash":
        if len(hash) == 0:
            msg = 'Must specify the "hash" columns to use md5_hash method.'
            raise ValueError(msg)

        data[to] = data.apply(lambda row: gen_md5_hash(row, hash), axis=1)
    # 数字ID
    elif method == "increment":
        data[to] = data.index + 1
    else:
        msg = f"Unknown method {method}."
        raise ValueError(msg)

    return TableContainer(table=data)
