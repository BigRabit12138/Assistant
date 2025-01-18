from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)

from assistant.memory.graphrag_v1.index.graph.extractors.community_reports.schemas import (
    CLAIM_ID,
    CLAIM_TYPE,
    CLAIM_STATUS,
    CLAIM_SUBJECT,
    CLAIM_DETAILS,
    CLAIM_DESCRIPTION,
)

_MISSING_DESCRIPTION = "No Description"


@verb(name="prepare_community_reports_claims")
def prepare_community_reports_claims(
        input: VerbInput,
        to: str = CLAIM_DETAILS,
        id_column: str = CLAIM_ID,
        description_column: str = CLAIM_DESCRIPTION,
        subject_column: str = CLAIM_SUBJECT,
        type_column: str = CLAIM_TYPE,
        status_column: str = CLAIM_STATUS,
        **_kwargs,
) -> TableContainer:
    claim_df: pd.DataFrame = cast(pd.DataFrame, input.get_input())
    claim_df = claim_df.fillna(value={description_column: _MISSING_DESCRIPTION})

    claim_df[to] = claim_df.apply(
        lambda x: {
            id_column: x[id_column],
            subject_column: x[subject_column],
            type_column: x[type_column],
            status_column: x[status_column],
            description_column: x[description_column],
        },
        axis=1,
    )

    return TableContainer(table=claim_df)
