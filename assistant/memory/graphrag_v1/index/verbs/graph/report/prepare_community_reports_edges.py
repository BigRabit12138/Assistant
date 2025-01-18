from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)

from assistant.memory.graphrag_v1.index.graph.extractors.community_reports.schemas import (
    EDGE_ID,
    EDGE_TARGET,
    EDGE_SOURCE,
    EDGE_DEGREE,
    EDGE_DETAILS,
    EDGE_DESCRIPTION,
)

_MISSING_DESCRIPTION = "No Description"


@verb(name="prepare_community_reports_edges")
def prepare_community_reports_edges(
        input: VerbInput,
        to: str = EDGE_DETAILS,
        id_column: str = EDGE_ID,
        source_column: str = EDGE_SOURCE,
        target_column: str = EDGE_TARGET,
        description_column: str = EDGE_DESCRIPTION,
        degree_column: str = EDGE_DEGREE,
        **_kwargs,
) -> TableContainer:
    """
    获取边的human_readable_id, source, target, description, rank属性字典，赋值给
    edge_details列
    :param input: 输入，包含表格
    :param to: 保存边属性的列
    :param id_column: 边id的列
    :param source_column: 边原点的列
    :param target_column: 边目的的列
    :param description_column: 边描述的列
    :param degree_column: 边的度
    :param _kwargs: 额外参数
    :return: 输出，包含表格
    """
    # 获取输入，description列NA填充"No Description"
    edge_df: pd.DataFrame = cast(pd.DataFrame, input.get_input()).fillna(
        value={description_column: _MISSING_DESCRIPTION}
    )
    # 获取边的human_readable_id, source, target, description, rank属性字典，赋值给
    # edge_details列
    edge_df[to] = edge_df.apply(
        lambda x: {
            id_column: x[id_column],
            source_column: x[source_column],
            target_column: x[target_column],
            description_column: x[description_column],
            degree_column: x[degree_column],
        },
        axis=1,
    )

    return TableContainer(table=edge_df)
