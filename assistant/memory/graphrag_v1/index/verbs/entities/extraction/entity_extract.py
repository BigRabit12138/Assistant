import logging

from enum import Enum
from typing import Any, cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    AsyncType,
    VerbCallbacks,
    TableContainer,
    derive_from_rows,
)

from assistant.memory.graphrag_v1.index.bootstrap import bootstrap
from assistant.memory.graphrag_v1.index.cache import PipelineCache

from assistant.memory.graphrag_v1.index.verbs.entities.extraction.strategies.typing import (
    Document,
    EntityExtractStrategy,
)

log = logging.getLogger(__name__)


class ExtractEntityStrategyType(str, Enum):
    """
    实体抽取的策略
    """
    graph_intelligence = "graph_intelligence"
    graph_intelligence_json = "graph_intelligence_json"
    nltk = "nltk"

    def __repr__(self):
        return f'"{self.value}"'


DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


@verb(name="entity_extract")
async def entity_extract(
        input: VerbInput,
        cache: PipelineCache,
        callbacks: VerbCallbacks,
        column: str,
        id_column: str,
        to: str,
        strategy: dict[str, Any] | None,
        graph_to: str | None = None,
        async_mode: AsyncType = AsyncType.AsyncIO,
        entity_types: list | None = None,
        **kwargs,
) -> TableContainer:
    """
    对输入表格的每一行的column列进行实体抽取，并将提取的实体和图附加到to和graph_to列
    :param input: 输入，包含输入表格
    :param cache: 缓存器
    :param callbacks: 回调函数
    :param column: 实体抽取处理的列
    :param id_column: id所在的列
    :param to: 保存实体的列
    :param strategy: 抽取策略
    :param graph_to: 保存实体图的列
    :param async_mode: 异步运行模式
    :param entity_types: 实体类型
    :param kwargs: 额外的参数
    :return: 输出，包含输出表格
    """
    log.debug(f"entity_extract strategy={strategy}")
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES

    output = cast(pd.DataFrame, input.get_input())
    strategy = strategy or {}
    # 获取抽取实体的方法
    strategy_exec = _load_strategy(
        strategy.get("type", ExtractEntityStrategyType.graph_intelligence)
    )
    strategy_config = {**strategy}

    num_started = 0

    async def run_strategy(row):
        """
        运行抽取策略
        :param row: 输入表格的行
        :return: 实体和实体图
        """
        nonlocal num_started
        text = row[column]
        id_ = row[id_column]
        # 运行抽取方法
        result_ = await strategy_exec(
            [Document(text=text, id=id_)],
            entity_types,
            callbacks,
            cache,
            strategy_config,
        )
        num_started += 1
        return [result_.entities, result_.graphml_graph]

    # 对所有行进行抽取实体
    results = await derive_from_rows(
        output,
        run_strategy,
        callbacks,
        scheduling_type=async_mode,
        num_threads=kwargs.get("num_threads", 4),
    )

    # 保存结果
    to_result = []
    graph_to_result = []
    for result in results:
        if result:
            to_result.append(result[0])
            graph_to_result.append(result[1])
        else:
            to_result.append(None)
            graph_to_result.append(None)

    output[to] = to_result
    if graph_to is not None:
        output[graph_to] = graph_to_result

    return TableContainer(table=output.reset_index(drop=True))


def _load_strategy(
        strategy_type: ExtractEntityStrategyType
) -> EntityExtractStrategy:
    """
    加载实体抽取策略
    :param strategy_type: 实体抽取策略
    :return: 实体抽取策略方法
    """
    match strategy_type:
        case ExtractEntityStrategyType.graph_intelligence:
            from assistant.memory.graphrag_v1.index.verbs.entities.extraction.strategies.graph_intelligence import run_gi

            return run_gi

        case ExtractEntityStrategyType.nltk:
            bootstrap()
            from assistant.memory.graphrag_v1.index.verbs.entities.extraction.strategies.nltk import run as run_nltk

            return run_nltk

        case _:
            msg = f"Unknown strategy: {strategy_type}"
            raise ValueError(msg)
