import logging

from enum import Enum
from typing import cast

import pandas as pd

from datashaper import (
    verb,
    AsyncType,
    VerbInput,
    VerbCallbacks,
    TableContainer,
    progress_ticker,
    derive_from_rows,
    NoopVerbCallbacks,
)

import assistant.memory.graphrag_v1.config.defaults as defaults
import assistant.memory.graphrag_v1.index.graph.extractors.community_reports.schemas as schemas

from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.utils.ds_util import get_required_input_table
from assistant.memory.graphrag_v1.index.graph.extractors.community_reports import (
    get_levels,
    prep_community_report_context,
)
from assistant.memory.graphrag_v1.index.verbs.graph.report.strategies.typing import (
    CommunityReport,
    CommunityReportsStrategy,
)

log = logging.getLogger(__name__)


class CreateCommunityReportsStrategyType(str, Enum):
    """
    总结聚簇的策略
    """
    graph_intelligence = "graph_intelligence"

    def __repr__(self):
        return f'"{self.value}"'


@verb(name="create_community_reports")
async def create_community_reports(
        input: VerbInput,
        callbacks: VerbCallbacks,
        cache: PipelineCache,
        strategy: dict,
        async_mode: AsyncType = AsyncType.AsyncIO,
        num_threads: int = 4,
        **_kwargs,
) -> TableContainer:
    """
    获取所有的聚簇的摘要结果
    :param input: 输入，包含表格
    :param callbacks: 回调钩子
    :param cache: 缓存器
    :param strategy: 摘要策略
    :param async_mode: 异步方式
    :param num_threads: 并发数
    :param _kwargs: 额外参数
    :return: 所有的聚簇的摘要结果
    """
    log.debug(f"create_community_reports strategy={strategy}.")
    local_contexts = cast(pd.DataFrame, input.get_input())
    nodes_ctr = get_required_input_table(input, "nodes")
    nodes = cast(pd.DataFrame, nodes_ctr.table)
    community_hierarchy_ctr = get_required_input_table(input, "community_hierarchy")
    community_hierarchy = cast(pd.DataFrame, community_hierarchy_ctr.table)

    # 获取节点所有的层
    levels = get_levels(nodes)
    reports: list[CommunityReport | None] = []
    tick = progress_ticker(callbacks.progress, len(local_contexts))
    # 加载总结策略
    runner = load_strategy(strategy["type"])

    # 获取所有的聚簇的摘要结果
    for level in levels:
        # 获取当前层的聚簇信息
        level_contexts = prep_community_report_context(
            pd.DataFrame(reports),
            local_context_df=local_contexts,
            community_hierarchy_df=community_hierarchy,
            level=level,
            max_tokens=strategy.get(
                "max_input_tokens", defaults.COMMUNITY_REPORT_MAX_INPUT_LENGTH
            ),
        )

        async def run_generate(record):
            """
            对每一个分层聚簇生成摘要
            :param record: 一个聚簇
            :return: 摘要文本
            """
            result = await _generate_report(
                runner,
                community_id=record[schemas.NODE_COMMUNITY],
                community_level=record[schemas.COMMUNITY_LEVEL],
                community_context=record[schemas.CONTEXT_STRING],
                cache=cache,
                callbacks=callbacks,
                strategy=strategy,
            )
            tick()
            return result
        # 获取当前层的所有聚簇的摘要结果
        local_reports = await derive_from_rows(
            level_contexts,
            run_generate,
            callbacks=NoopVerbCallbacks(),
            num_threads=num_threads,
            scheduling_type=async_mode,
        )
        # 去掉空的
        reports.extend([lr for lr in local_reports if lr is not None])

    return TableContainer(table=pd.DataFrame(reports))


async def _generate_report(
        runner: CommunityReportsStrategy,
        cache: PipelineCache,
        callbacks: VerbCallbacks,
        strategy: dict,
        community_id: int | str,
        community_level: int,
        community_context: str,
) -> CommunityReport | None:
    """
    对一个聚簇的对象的所有信息调用大模型摘要总结
    :param runner: 摘要方法
    :param cache: 缓存器
    :param callbacks: 回调钩子
    :param strategy: 大模型参数
    :param community_id: community列
    :param community_level: level列
    :param community_context: 文本列
    :return: 摘要内容
    """
    return await runner(
        community_id,
        community_context,
        community_level,
        callbacks,
        cache,
        strategy
    )


def load_strategy(
        strategy: CreateCommunityReportsStrategyType,
) -> CommunityReportsStrategy:
    """
    加载总结聚簇的策略方法
    :param strategy: 策略类型
    :return: 总结聚簇的策略方法
    """
    match strategy:
        case CreateCommunityReportsStrategyType.graph_intelligence:
            from assistant.memory.graphrag_v1.index.verbs.graph.report.strategies.graph_intelligence import run

            return run

        case _:
            msg = f"Unknown strategy: {strategy}."
            raise ValueError(msg)
