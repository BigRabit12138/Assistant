from typing import Any, cast

import pandas as pd

from assistant.memory.graphrag_v1.model import (
    Entity,
    TextUnit,
)


def get_candidate_text_units(
        selected_entities: list[Entity],
        text_units: list[TextUnit],
) -> pd.DataFrame:
    selected_text_ids = [
        entity.text_unit_ids for entity in selected_entities if entity.text_unit_ids
    ]
    selected_text_ids = [item for sublist in selected_text_ids for item in sublist]
    selected_text_units = [unit for unit in text_units if unit.id in selected_text_ids]
    return to_text_unit_dataframe(selected_text_units)


def to_text_unit_dataframe(
        text_units: list[TextUnit]
) -> pd.DataFrame:
    if len(text_units) == 0:
        return pd.DataFrame()

    header = ["id", "text"]
    attribute_cols = (
        list(text_units[0].attributes.keys()) if text_units[0].attributes else []
    )
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)

    records = []
    for unit in text_units:
        new_record = [
            unit.short_id,
            unit.text,
            *[
                str(unit.attributes.get(field, ""))
                if unit.attributes and unit.attributes.get(field)
                else ""
                for field in attribute_cols
            ],
        ]
        records.append(new_record)
    return pd.DataFrame(records, columns=cast(Any, header))
