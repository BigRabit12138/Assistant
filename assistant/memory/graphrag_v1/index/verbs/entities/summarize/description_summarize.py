import asyncio
import logging

from enum import Enum
from typing import Any, NamedTuple, cast

import pandas as pd
import networkx as nx

from datashaper import (
    verb,
    VerbInput,
    VerbCallbacks,
    TableContainer,
    ProgressTicker,
    progress_ticker,
)

from assistant.memory.graphrag_v1.index.utils import load_graph
from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.verbs.entities.summarize.strategies.typing import (
    SummarizationStrategy,
    SummarizedDescriptionResult,
)


log = logging.getLogger(__name__)


class DescriptionSummarizeRow(NamedTuple):
    """
    摘要结果
    """
    graph: Any


class SummarizeStrategyType(str, Enum):
    """
    摘要策略类型
    """
    graph_intelligence = "graph_intelligence"

    def __repr__(self):
        return f'"{self.value}"'


@verb(name="summarize_descriptions")
async def summarize_descriptions(
        input: VerbInput,
        cache: PipelineCache,
        callbacks: VerbCallbacks,
        column: str,
        to: str,
        strategy: dict[str, Any] | None = None,
        **kwargs,
) -> TableContainer:
    """
    对column列的图的图对象进行摘要，并将新图并赋值给to列
    :param input: 输入，包含输入表格
    :param cache: 大模型缓存器
    :param callbacks: 进度条
    :param column: 原始图所在列
    :param to: 摘要后的图所在列
    :param strategy: 摘要策略
    :param kwargs: 额外参数
    :return: 输出，包含输出表格
    """
    log.debug(f"summarize_descriptions strategy={strategy}.")
    output = cast(pd.DataFrame, input.get_input())
    strategy = strategy or {}
    # 加载摘要方法
    strategy_exec = load_strategy(
        strategy.get("type", SummarizeStrategyType.graph_intelligence)
    )
    strategy_config = {**strategy}

    async def get_resolved_entities(
            row,
            semaphore_: asyncio.Semaphore
    ) -> DescriptionSummarizeRow:
        """
        对一个图的所有对象进行摘要
        :param row: 表格的每一行，每行有一个图
        :param semaphore_: 信号量
        :return: 摘要结果
        """
        # 加载图
        graph: nx.Graph = load_graph(cast(str | nx.Graph, getattr(row, column)))

        ticker_length = len(graph.nodes) + len(graph.edges)
        ticker = progress_ticker(callbacks.progress, ticker_length)

        # 摘要节点
        futures = [
            do_summarize_descriptions(
                node,
                # 获取全部描述，去重并排序
                sorted(set(graph.nodes[node].get("description", "").split("\n"))),
                ticker,
                semaphore_,
            )
            for node in graph.nodes()
        ]
        # 摘要边
        futures += [
            do_summarize_descriptions(
                edge,
                sorted(set(graph.edges[edge].get("description", "").split("\n"))),
                ticker,
                semaphore_,
            )
            for edge in graph.edges()
        ]
        # 获取所有的摘要结果
        results_ = await asyncio.gather(*futures)

        for result_ in results_:
            graph_item = result_.items
            # 覆盖节点原来的描述
            if isinstance(graph_item, str) and graph_item in graph.nodes():
                graph.nodes[graph_item]["description"] = result_.description
            # 覆盖边原来的描述
            elif isinstance(graph_item, tuple) and graph_item in graph.edges():
                graph.edges[graph_item]["description"] = result_.description

        return DescriptionSummarizeRow(
            graph="\n".join(nx.generate_graphml(graph)),
        )

    async def do_summarize_descriptions(
            graph_item: str | tuple[str, str],
            descriptions: list[str],
            ticker: ProgressTicker,
            semaphore_: asyncio.Semaphore,
    ) -> SummarizedDescriptionResult:
        """
        摘要一个图对象的描述
        :param graph_item: 图对象key
        :param descriptions: 图对象描述
        :param ticker: 进度条
        :param semaphore_: 信号量
        :return: 摘要结果
        """
        async with semaphore_:
            results_ = await strategy_exec(
                graph_item,
                descriptions,
                callbacks,
                cache,
                strategy_config,
            )
            ticker(1)
        return results_

    semaphore = asyncio.Semaphore(kwargs.get("num_threads", 4))
    # 对所有图的描述进行摘要，并覆盖
    results = [
        await get_resolved_entities(row, semaphore) for row in output.itertuples()
    ]

    # 构造新列
    to_result = []
    for result in results:
        if result:
            to_result.append(result.graph)
        else:
            to_result.append(None)
    output[to] = to_result

    return TableContainer(table=output)


def load_strategy(
        strategy_type: SummarizeStrategyType
) -> SummarizationStrategy:
    """
    加载摘要节点的策略
    :param strategy_type: 摘要策略类型
    :return: 摘要策略
    """
    match strategy_type:
        case SummarizeStrategyType.graph_intelligence:
            from assistant.memory.graphrag_v1.index.verbs.entities.summarize.strategies.graph_intelligence \
                import run as run_gi

            return run_gi
        case _:
            msg = f"Unknown strategy: {strategy_type}."
            raise ValueError(msg)
