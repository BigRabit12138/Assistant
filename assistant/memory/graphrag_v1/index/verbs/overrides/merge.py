import logging

from enum import Enum
from typing import Any, cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    VerbResult,
    TableContainer,
)
from datashaper.engine.verbs.merge import merge as ds_merge


log = logging.getLogger(__name__)


class MergeStrategyType(str, Enum):
    json = "json"
    datashaper = "datashaper"

    def __repr__(self):
        return f'"{self.value}"'


@verb(name="merge_override")
def merge(
        input: VerbInput,
        to: str,
        columns: list[str],
        strategy: MergeStrategyType = MergeStrategyType.datashaper,
        delimiter: str = '',
        preserve_source: bool = False,
        un_hot: bool = False,
        prefix: str = "",
        **_kwargs: dict,
) -> TableContainer | VerbResult:
    output: pd.DataFrame
    match strategy:
        case MergeStrategyType.json:
            output = _merge_json(input, to, columns)
            filtered_list: list[str] = []

            for col in output.columns:
                try:
                    columns.index(col)
                except ValueError:
                    log.exception(f"Column {col} not found in input columns.")
                filtered_list.append(col)

            if not preserve_source:
                output = cast(Any, output[filtered_list])
            return TableContainer(table=output.reset_index())
        case _:
            return ds_merge(
                input,
                to,
                columns,
                strategy,
                delimiter,
                preserve_source,
                un_hot,
                prefix
            )


def _merge_json(
        input_: VerbInput,
        to: str,
        columns: list[str],
) -> pd.DataFrame:
    input_table = cast(pd.DataFrame, input_.get_input())
    output = input_table
    output[to] = output[columns].apply(
        lambda row: ({**row}),
        axis=1,
    )
    return output
