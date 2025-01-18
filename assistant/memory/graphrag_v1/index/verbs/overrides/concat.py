from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)


@verb(name="concat_override")
def concat(
        input: VerbInput,
        column_wise: bool = False,
        **_kwargs: dict,
) -> TableContainer:
    input_table = cast(pd.DataFrame, input.get_input())
    others = cast(list[pd.DataFrame], input.get_others())
    if column_wise:
        output = pd.concat([input_table, *others], axis=1)
    else:
        output = pd.concat([input_table, *others], ignore_index=True)

    return TableContainer(table=output)
