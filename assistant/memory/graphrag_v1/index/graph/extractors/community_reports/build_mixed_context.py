import pandas as pd

import assistant.memory.graphrag_v1.index.graph.extractors.community_reports.schemas as schemas

from assistant.memory.graphrag_v1.query.llm.text_utils import num_tokens
from assistant.memory.graphrag_v1.index.graph.extractors.community_reports.sort_context import sort_context


def build_mixed_context(
        context: list[dict],
        max_tokens: int
) -> str:
    """
    获取子聚簇的文本信息
    :param context: 一个聚簇的所有context
    :param max_tokens: 最大token数量
    :return: 子聚簇的文本信息
    """
    # TODO: 太鸡儿难了，没看懂
    # 按context_size对all_context倒序排列
    sorted_context = sorted(
        context, key=lambda x: x[schemas.CONTEXT_SIZE], reverse=True
    )

    substitute_reports = []
    final_local_contexts = []
    exceeded_limit = True
    context_string = ""

    # 从所有子聚簇中获取一个长度合格的子聚簇的文本信息
    for idx, sub_community_context in enumerate(sorted_context):
        if exceeded_limit:
            if sub_community_context[schemas.FULL_CONTENT]:
                substitute_reports.append(
                    {
                        schemas.COMMUNITY_ID: sub_community_context[schemas.SUB_COMMUNITY],
                        schemas.FULL_CONTENT: sub_community_context[schemas.FULL_CONTENT],
                    }
                )
            else:
                final_local_contexts.extend(sub_community_context[schemas.ALL_CONTEXT])
                continue

            remaining_local_context = []
            for rid in range(idx + 1, len(sorted_context)):
                remaining_local_context.extend(sorted_context[rid][schemas.ALL_CONTEXT])
            # 获取子聚簇的文本信息
            new_context_string = sort_context(
                local_context=remaining_local_context + final_local_contexts,
                sub_community_reports=substitute_reports,
            )
            if num_tokens(new_context_string) <= max_tokens:
                exceeded_limit = False
                context_string = new_context_string
                break

    if exceeded_limit:
        substitute_reports = []
        for sub_community_context in sorted_context:
            substitute_reports.append(
                {
                    schemas.COMMUNITY_ID: sub_community_context[schemas.SUB_COMMUNITY],
                    schemas.FULL_CONTENT: sub_community_context[schemas.FULL_CONTENT],
                }
            )
            new_context_string = pd.DataFrame(substitute_reports).to_csv(
                index=False, sep=","
            )
            if num_tokens(new_context_string) > max_tokens:
                break

            context_string = new_context_string

    return context_string
