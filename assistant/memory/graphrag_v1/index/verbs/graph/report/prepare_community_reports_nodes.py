from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)

from assistant.memory.graphrag_v1.index.graph.extractors.community_reports.schemas import (
    NODE_ID,
    NODE_NAME,
    NODE_DEGREE,
    NODE_DETAILS,
    NODE_DESCRIPTION,
)

_MISSING_DESCRIPTION = "No Description"


@verb(name="prepare_community_reports_nodes")
def prepare_community_reports_nodes(
        input: VerbInput,
        to: str = NODE_DETAILS,
        id_column: str = NODE_ID,
        name_column: str = NODE_NAME,
        description_column: str = NODE_DESCRIPTION,
        degree_column: str = NODE_DEGREE,
        **_kwargs,
) -> TableContainer:
    """
    提取出节点的human_readable_id, title, description, degree属性
    赋值给node_details列
    :param input: 输入，包含表格
    :param to: 保存节点属性的列
    :param id_column: 节点id的列
    :param name_column: 节点名称的列
    :param description_column: 节点描述的列
    :param degree_column: 节点度的列
    :param _kwargs: 额外参数
    :return: 输出，包含表格
    """
    node_df: pd.DataFrame = cast(pd.DataFrame, input.get_input())
    # 把description列的NA填充"No Description"
    node_df = node_df.fillna(
        value={description_column: _MISSING_DESCRIPTION}
    )
    # 提取出节点的human_readable_id, title, description, degree属性字典，
    # 赋值给node_details列
    node_df[to] = node_df.apply(
        lambda x: {
            id_column: x[id_column],
            name_column: x[name_column],
            description_column: x[description_column],
            degree_column: x[degree_column],
        },
        axis=1,
    )

    return TableContainer(table=node_df)
