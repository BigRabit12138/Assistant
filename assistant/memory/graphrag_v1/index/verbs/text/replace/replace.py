from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)

from assistant.memory.graphrag_v1.index.verbs.text.replace.typing import Replacement


@verb(name="text_replace")
def text_replace(
        input: VerbInput,
        column: str,
        to: str,
        replacements: list[dict[str, str]],
        **_kwargs: dict,
) -> TableContainer:
    output = cast(pd.DataFrame, input.get_input())
    parsed_replacements = [Replacement(**r) for r in replacements]
    output[to] = output[column].apply(
        lambda text: _apply_replacements(text, parsed_replacements)
    )
    return TableContainer(table=output)


def _apply_replacements(
        text: str,
        replacements: list[Replacement]
) -> str:
    for r in replacements:
        text = text.replace(r.pattern, r.replacement)
    return text
