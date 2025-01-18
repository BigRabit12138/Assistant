import logging

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)

from assistant.memory.graphrag_v1.index.utils import is_null

DEFAULT_COPY = ["level"]


@verb(name="spread_json")
def spread_json(
        input: VerbInput,
        column: str,
        copy: list[str] | None = None,
        **_kwargs: dict,
) -> TableContainer:
    if copy is None:
        copy = DEFAULT_COPY
    data = input.get_input()

    results = []

    for _, row in data.iterrows():
        try:
            cleaned_row = {col: row[col] for col in copy}
            rest_row = row[column] if row[column] is not None else {}

            if is_null(rest_row):
                rest_row = {}
            results.append({**cleaned_row, **rest_row})
        except Exception:
            logging.exception(f"Error spreading row: {row}.")
            raise

    data = pd.DataFrame(results, index=data.index)

    return TableContainer(table=data)
