import logging

from typing import cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)

import assistant.memory.graphrag_v1.index.graph.extractors.community_reports.schemas as schemas

log = logging.getLogger(__name__)


@verb(name="restore_community_hierarchy")
def restore_community_hierarchy(
        input: VerbInput,
        name_column: str = schemas.NODE_NAME,
        community_column: str = schemas.NODE_COMMUNITY,
        level_column: str = schemas.NODE_LEVEL,
        **_kwargs,
) -> TableContainer:
    """
    找到图中节点分层聚簇中，层间的所有community的包含关系
    :param input: 输入，包含表格
    :param name_column: 节点的名字所在列
    :param community_column: 节点的community所在列
    :param level_column: 节点的level所在列
    :param _kwargs: 额外的参数
    :return: 输出，包含表格
    """
    node_df: pd.DataFrame = cast(pd.DataFrame, input.get_input())
    # 按community, level列分组，将title列聚合成列表，并重置索引
    community_df = (
        node_df.groupby([community_column, level_column])
        .agg({name_column: list})
        .reset_index()
    )
    # 获取节点的分层聚簇信息
    # {level: {community: node_name}}
    community_levels = {}
    for _, row in community_df.iterrows():
        level = row[level_column]
        name = row[name_column]
        community = row[community_column]

        if community_levels.get(level) is None:
            community_levels[level] = {}
        community_levels[level][community] = name

    # 有序levels
    levels = sorted(community_levels.keys())

    community_hierarchy = []
    # 找到所有层之间的包含关系
    # TODO: 以后记得看
    for idx in range(len(levels) - 1):
        level = levels[idx]
        log.debug(f"Level: {level}.")
        next_level = levels[idx + 1]
        current_level_communities = community_levels[level]
        next_level_communities = community_levels[next_level]
        log.debug(
            f"Number of communities at level {level}: {len(current_level_communities)}"
        )
        # 找到当前层对下一次可能的包含关系
        for current_community in current_level_communities:
            current_entities = current_level_communities[current_community]

            entities_found = 0
            # 找到当前community对下一层可能的包含关系
            for next_level_community in next_level_communities:
                next_entities = next_level_communities[next_level_community]
                # 如果下一个层的community是当前community的子集
                # 记录下来这个包含关系
                if set(next_entities).issubset(set(current_entities)):
                    community_hierarchy.append(
                        {
                            community_column: current_community,
                            schemas.COMMUNITY_LEVEL: level,
                            schemas.SUB_COMMUNITY: next_level_community,
                            schemas.SUB_COMMUNITY_SIZE: len(next_entities),

                        }
                    )

                    entities_found += len(next_entities)
                    # 当前community的所有节点都在下一次中被找到，结束对当前community的搜索
                    if entities_found == len(current_entities):
                        break

    return TableContainer(table=pd.DataFrame(community_hierarchy))
