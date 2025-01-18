import logging

from enum import Enum
from typing import Any, cast

import numpy as np
import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    VerbCallbacks,
    TableContainer,
)

from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.verbs.text.embed.strategies.typing import TextEmbeddingStrategy
from assistant.memory.graphrag_v1.vector_stores import (
    BaseVectorStore,
    VectorStoreFactory,
    VectorStoreDocument,
)

log = logging.getLogger(__name__)

DEFAULT_EMBEDDING_BATCH_SIZE = 500


class TextEmbedStrategyType(str, Enum):
    """
    embedding策略
    """
    openai = "openai"
    mock = "mock"

    def __repr__(self):
        return f'"{self.value}"'


@verb(name="text_embed")
async def text_embed(
        input: VerbInput,
        callbacks: VerbCallbacks,
        cache: PipelineCache,
        column: str,
        strategy: dict,
        **kwargs,
) -> TableContainer:
    """
    对输入文本作embedding
    :param input: 输入，包含输入表格
    :param callbacks: 回调钩子
    :param cache: 缓存器
    :param column: 文本所在列
    :param strategy: embedding策略
    :param kwargs: embedding配置
    :return: 输出，包含输出表格
    """
    vector_store_config = strategy.get("vector_store")

    # 使用矢量数据库缓存embeddings
    if vector_store_config:
        embedding_name = kwargs.get("embedding_name", "default")
        collection_name = _get_collection_name(vector_store_config, embedding_name)
        vector_store: BaseVectorStore = _create_vector_store(
            vector_store_config, collection_name
        )
        vector_store_workflow_config = vector_store_config.get(
            embedding_name, vector_store_config
        )
        return await _text_embed_with_vector_store(
            input,
            callbacks,
            cache,
            column,
            strategy,
            vector_store,
            vector_store_workflow_config,
            vector_store_config.get("store_in_table", False),
            kwargs.get("to", f"{column}_embedding"),
        )
    # 在内存中缓存embeddings
    return await _text_embed_in_memory(
        input,
        callbacks,
        cache,
        column,
        strategy,
        kwargs.get("to", f"{column}_embedding"),
    )


async def _text_embed_in_memory(
        input_: VerbInput,
        callbacks: VerbCallbacks,
        cache: PipelineCache,
        column: str,
        strategy: dict,
        to: str,
) -> TableContainer:
    """
    对输入文本作embedding
    :param input_: 输入，包含输入表格
    :param callbacks: 回调钩子
    :param cache: 缓存器
    :param column: 文本所在列
    :param strategy: embedding策略
    :param to: 保存embedding的列
    :return: 输出，包含输出表格
    """
    output_df = cast(pd.DataFrame, input_.get_input())
    # 加载embedding策略
    strategy_type = strategy["type"]
    strategy_exec = load_strategy(strategy_type)
    strategy_args = {**strategy}

    input_table = input_.get_input()

    # 获取需要embedding的文本
    texts: list[str] = input_table[column].to_numpy().tolist()
    # embedding文本
    result = await strategy_exec(texts, callbacks, cache, strategy_args)

    # 结果赋值给to列
    output_df[to] = result.embeddings
    return TableContainer(table=output_df)


async def _text_embed_with_vector_store(
        input_: VerbInput,
        callbacks: VerbCallbacks,
        cache: PipelineCache,
        column: str,
        strategy: dict[str, Any],
        vector_store: BaseVectorStore,
        vector_store_config: dict,
        store_in_table: bool = False,
        to: str = "",
):
    output_df = cast(pd.DataFrame, input_.get_input())
    strategy_type = strategy["type"]
    strategy_exec = load_strategy(strategy_type)
    strategy_args = {**strategy}

    insert_batch_size: int = (
        vector_store_config.get("batch_size") or DEFAULT_EMBEDDING_BATCH_SIZE
    )
    title_column: str = vector_store_config.get("title_column", "title")
    id_column: str = vector_store_config.get("id_column", "id")
    overwrite: bool = vector_store_config.get("overwrite", True)

    if column not in output_df.columns:
        msg = f"Column {column} not found in input dataframe with columns {output_df.columns}."
        raise ValueError(msg)

    if title_column not in output_df.columns:
        msg = f"Column {title_column} not found in input dataframe with columns {output_df.columns}."
        raise ValueError(msg)
    if id_column not in output_df.columns:
        msg = f"Column {id_column} not found in input dataframe with columns {output_df.columns}."
        raise ValueError(msg)

    total_rows = 0
    for row in output_df[column]:
        if isinstance(row, list):
            total_rows += len(row)
        else:
            total_rows += 1

    i = 0
    starting_index = 0

    all_results = []

    while insert_batch_size * i < input_.get_input().shape[0]:
        batch = input_.get_input().iloc[
            insert_batch_size * i: insert_batch_size * (i + 1)
        ]
        texts: list[str] = batch[column].to_numpy().tolist()
        titles: list[str] = batch[title_column].to_numpy().tolist()
        ids: list[str] = batch[id_column].to_numpy().tolist()
        result = await strategy_exec(
            texts,
            callbacks,
            cache,
            strategy_args,
        )
        if store_in_table and result.embeddings:
            embeddings = [
                embedding for embedding in result.embeddings if embedding is not None
            ]
            all_results.extend(embeddings)

        vectors = result.embeddings or []
        documents: list[VectorStoreDocument] = []
        for id_, text, title, vector in zip(ids, texts, titles, vectors, strict=True):
            if type(vector) is np.ndarray:
                vector = vector.tolist()
            document = VectorStoreDocument(
                id=id_,
                text=text,
                vector=vector,
                attributes={"title": title},
            )
            documents.append(document)

        vector_store.load_documents(documents, overwrite and i == 0)
        starting_index += len(documents)
        i += 1

    if store_in_table:
        output_df[to] = all_results

    return TableContainer(table=output_df)


def _create_vector_store(
        vector_store_config: dict,
        collection_name: str,
) -> BaseVectorStore:
    vector_store_type: str = str(vector_store_config.get("type"))
    if collection_name:
        vector_store_config.update({"collection_name": collection_name})
    vector_store = VectorStoreFactory.get_vector_store(
        vector_store_type, kwargs=vector_store_config
    )
    vector_store.connect(**vector_store_config)
    return vector_store


def _get_collection_name(
        vector_store_config: dict,
        embedding_name: str
) -> str:
    collection_name = vector_store_config.get("collection_name")
    if not collection_name:
        collection_names = vector_store_config.get("collection_names", {})
        collection_name = collection_names.get(embedding_name, embedding_name)

    msg = f"using {vector_store_config.get('type')} collection_name {collection_name} for embedding {embedding_name}."
    log.info(msg)
    return collection_name


def load_strategy(
        strategy: TextEmbedStrategyType
) -> TextEmbeddingStrategy:
    """
    加载embedding策略
    :param strategy: embedding策略
    :return: embedding方法
    """
    match strategy:
        case TextEmbedStrategyType.openai:
            from assistant.memory.graphrag_v1.index.verbs.text.embed.strategies.openai import run as run_openai

            return run_openai
        case TextEmbedStrategyType.mock:
            from assistant.memory.graphrag_v1.index.verbs.text.embed.strategies.mock import run as run_mock

            return run_mock
        case _:
            msg = f"Unknown strategy: {strategy}."
            raise ValueError(msg)
