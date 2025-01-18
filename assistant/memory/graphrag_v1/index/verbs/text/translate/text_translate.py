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

from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.verbs.text.translate.strategies.typing import TextTranslationStrategy


class TextTranslateStrategyType(str, Enum):
    openai = "openai"
    mock = "mock"

    def __repr__(self):
        return f'"{self.value}"'


@verb(name="text_translate")
async def text_translate(
        input: VerbInput,
        cache: PipelineCache,
        callbacks: VerbCallbacks,
        text_column: str,
        to: str,
        strategy: dict[str, Any],
        async_mode: AsyncType = AsyncType.AsyncIO,
        **kwargs,
) -> TableContainer:
    output_df = cast(pd.DataFrame, input.get_input())
    strategy_type = strategy["type"]
    strategy_args = {**strategy}
    strategy_exec = _load_strategy(strategy_type)

    async def run_strategy(row):
        text = row[text_column]
        result = await strategy_exec(text, strategy_args, callbacks, cache)

        if isinstance(text, str):
            return result.translations[0]

        return list(result.translations)

    results = await derive_from_rows(
        output_df,
        run_strategy,
        callbacks,
        scheduling_type=async_mode,
        num_threads=kwargs.get("num_threads", 4),
    )
    output_df[to] = results
    return TableContainer(table=output_df)


def _load_strategy(
        strategy: TextTranslateStrategyType
) -> TextTranslationStrategy:
    match strategy:
        case TextTranslateStrategyType.openai:
            from assistant.memory.graphrag_v1.index.verbs.text.translate.strategies.openai import run as run_openai

            return run_openai
        case TextTranslateStrategyType.mock:
            from assistant.memory.graphrag_v1.index.verbs.text.translate.strategies.mock import run as run_mock

            return run_mock
        case _:
            msg = f"Unknown strategy: {strategy}."
            raise ValueError(msg)
