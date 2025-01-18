from enum import Enum
from typing import Any, cast

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    VerbCallbacks,
    TableContainer,
    ProgressTicker,
    progress_ticker,
)

from assistant.memory.graphrag_v1.index.verbs.text.chunk.typing import ChunkInput
from assistant.memory.graphrag_v1.index.verbs.text.chunk.strategies.typing import ChunkStrategy as ChunkStrategy


def _get_num_total(output: pd.DataFrame, column: str) -> int:
    """
    获取总共输入的文档数
    :param output: 输入表格
    :param column: 文本所在的列
    :return: 所有文档的数量
    """
    num_total = 0
    for row in output[column]:
        if isinstance(row, str):
            num_total += 1
        else:
            num_total += len(row)
    return num_total


class ChunkStrategyType(str, Enum):
    """
    切分文本策略类型
    """
    tokens = "tokens"
    sentence = "sentence"

    def __repr__(self):
        return f'"{self.value}"'


@verb(name="chunk")
def chunk(
        input: VerbInput,
        column: str,
        to: str,
        callbacks: VerbCallbacks,
        strategy: dict[str, Any] | None = None,
        **_kwargs,
) -> TableContainer:
    """
    对输入表格的column列的文本进行切分，并将结果赋给新列to
    :param input: 输入，包含输入表格
    :param column: 切分文本所在列
    :param to: 文本块列
    :param callbacks: 回调钩子
    :param strategy: 切分策略
    :param _kwargs: 额外参数
    :return: 输出，包含输出表格
    """
    if strategy is None:
        strategy = {}

    output = cast(pd.DataFrame, input.get_input())

    # 获取切分策略
    strategy_name = strategy.get("type", ChunkStrategyType.tokens)
    strategy_config = {**strategy}
    # 加载策略函数
    strategy_exec = load_strategy(strategy_name)

    num_total = _get_num_total(output, column)
    tick = progress_ticker(callbacks.progress, num_total)

    # 切分并将结果添加到新列
    output[to] = output.apply(
        cast(
            Any,
            lambda x: run_strategy(
                strategy_exec,
                x[column],
                strategy_config,
                tick,
            ),
        ),
        axis=1,
    )
    return TableContainer(table=output)


def run_strategy(
        strategy: ChunkStrategy,
        input_: ChunkInput,
        strategy_args: dict[str, Any],
        tick: ProgressTicker,
) -> list[str | tuple[list[str] | None, str, int]]:
    """
    对表格的一个指定单元进行文本分块
    :param strategy: 切分方法
    :param input_: 每行的表格单元
    :param strategy_args: 切分策略参数
    :param tick: 进度条
    :return: 切分的文本块
    """
    # 表格元素是文本
    if isinstance(input_, str):
        return [item.text_chunk for item in strategy([input_], {**strategy_args}, tick)]

    # 获取输入中的文本
    texts = []
    for item in input_:
        if isinstance(item, str):
            texts.append(item)
        else:
            texts.append(item[1])

    strategy_results = strategy(texts, {**strategy_args}, tick)

    results = []
    for strategy_result in strategy_results:
        doc_indices = strategy_result.source_doc_indices
        # 输入是list[str]的情况
        if isinstance(input_[doc_indices[0]], str):
            results.append(strategy_result.text_chunk)
        # 输入是list[tuple[str, str]]的情况，文本带有id
        else:
            doc_ids = [input_[doc_idx][0] for doc_idx in doc_indices]
            results.append((
                doc_ids,
                strategy_result.text_chunk,
                strategy_result.n_tokens,
            ))

    return results


def load_strategy(strategy: ChunkStrategyType) -> ChunkStrategy:
    """
    获取切分文本的策略，可执行函数
    :param strategy: 策略类型
    :return: 策略
    """
    match strategy:
        case ChunkStrategyType.tokens:
            from assistant.memory.graphrag_v1.index.verbs.text.chunk.strategies.tokens import run as run_tokens

            return run_tokens

        case ChunkStrategyType.sentence:
            from assistant.memory.graphrag_v1.index.bootstrap import bootstrap
            from assistant.memory.graphrag_v1.index.verbs.text.chunk.strategies.sentence import run as run_sentence

            bootstrap()
            return run_sentence

        case _:
            msg = f"Unknown strategy: {strategy}."
            raise ValueError(msg)
