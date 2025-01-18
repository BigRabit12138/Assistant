import asyncio
import logging

from typing import Any

import numpy as np

from datashaper import (
    VerbCallbacks,
    ProgressTicker,
    progress_ticker,
)

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.index.utils import is_null
from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.llm import load_llm_embeddings
from assistant.memory.graphrag_v1.llm import EmbeddingLLM, OpenAIConfiguration
from assistant.memory.graphrag_v1.index.text_splitting import TokenTextSplitter
from assistant.memory.graphrag_v1.index.verbs.text.embed.strategies.typing import TextEmbeddingResult

log = logging.getLogger(__name__)


async def run(
        input_: list[str],
        callbacks: VerbCallbacks,
        cache: PipelineCache,
        args: dict[str, Any],
) -> TextEmbeddingResult:
    """
    调用embedding模型对文本embedding
    :param input_: 输入文本
    :param callbacks: 回调钩子
    :param cache: 缓存器
    :param args: embedding配置参数
    :return: embedding结果
    """
    if is_null(input_):
        return TextEmbeddingResult(embeddings=None)

    llm_config = args.get("llm", {})
    batch_size = args.get("batch_size", 16)
    batch_max_tokens = args.get("batch_max_tokens", 8191)
    oai_config = OpenAIConfiguration(llm_config)
    # 获取按token的文本分割器
    splitter = _get_splitter(oai_config, batch_max_tokens)
    # 获取embedding模型
    llm = _get_llm(oai_config, callbacks, cache)
    semaphore: asyncio.Semaphore = asyncio.Semaphore(args.get("num_threads", 4))
    # 分割文本
    texts, input_sizes = _prepare_embed_texts(input_, splitter)
    # 将所有文本分批
    text_batches = _create_text_batches(
        texts,
        batch_size,
        batch_max_tokens,
        splitter,
    )
    log.info(
        f"embedding {len(input_)} inputs via {len(texts)} snippets using {len(text_batches)} batches"
        f". max_batch_size={batch_size}, max_tokens={batch_max_tokens}"
    )
    ticker = progress_ticker(callbacks.progress, len(text_batches))
    # 获取每一块的embedding
    embeddings = await _execute(llm, text_batches, ticker, semaphore)
    # 获取每一块原始文本的embedding
    embeddings = _reconstitute_embeddings(embeddings, input_sizes)

    return TextEmbeddingResult(embeddings=embeddings)


def _get_splitter(
        config: OpenAIConfiguration, batch_max_tokens: int
) -> TokenTextSplitter:
    """
    获取文本分割器
    :param config: openai大模型配置
    :param batch_max_tokens: 块token数量
    :return: 分割器
    """
    return TokenTextSplitter(
        encoding_name=config.encoding_model or defaults.ENCODING_MODEL,
        chunk_size=batch_max_tokens,
    )


def _get_llm(
        config: OpenAIConfiguration,
        callbacks: VerbCallbacks,
        cache: PipelineCache,
) -> EmbeddingLLM:
    """
    获取embedding模型
    :param config: 模型配置
    :param callbacks: 回调钩子
    :param cache: 缓存器
    :return: embedding模型
    """
    llm_type = config.lookup("type", "Unknown")
    return load_llm_embeddings(
        "text_embedding",
        llm_type,
        callbacks,
        cache,
        config.raw_config,
    )


async def _execute(
        llm: EmbeddingLLM,
        chunks: list[list[str]],
        tick: ProgressTicker,
        semaphore: asyncio.Semaphore,
) -> list[list[float]]:
    """
    调用embedding模型回去embedding
    :param llm: embedding模型
    :param chunks: 需要embedding的文本
    :param tick: 进度条
    :param semaphore: 并发控制器
    :return: embedding
    """
    async def embed(
            chunk: list[str]
    ) -> np.array:
        """
        对一批文本进行embedding
        :param chunk: 一批文本
        :return: 一批文本的embedding
        """
        async with semaphore:
            chunk_embeddings = await llm(chunk)
            result = np.array(chunk_embeddings.output)
            tick(1)
        return result

    # 对每一批进行embedding
    futures = [embed(chunk) for chunk in chunks]
    results = await asyncio.gather(*futures)
    # 展开，消掉批这一层
    return [item for sublist in results for item in sublist]


def _create_text_batches(
        texts: list[str],
        max_batch_size: int,
        max_batch_tokens: int,
        splitter: TokenTextSplitter,
) -> list[list[str]]:
    """
    将所有文本分批
    :param texts: 所有文本
    :param max_batch_size: 一批最大块数
    :param max_batch_tokens: 一批最大token
    :param splitter: 分割器
    :return: 分批的文本
    """
    result = []
    current_batch = []
    current_batch_tokens = 0

    for text in texts:
        token_count = splitter.num_tokens(text)
        # 获得一个batch，并重置当前batch内容
        if (
            len(current_batch) >= max_batch_size
            or current_batch_tokens + token_count > max_batch_tokens
        ):
            result.append(current_batch)
            current_batch = []
            current_batch_tokens = 0

        current_batch.append(text)
        current_batch_tokens += token_count
    # 处理尾巴
    if len(current_batch) > 0:
        result.append(current_batch)

    return result


def _prepare_embed_texts(
        input_: list[str],
        splitter: TokenTextSplitter
) -> tuple[list[str], list[int]]:
    """
    将所有文本按token分块
    :param input_: 输入文本
    :param splitter: 文本分割器
    :return: 分块的文本
    """
    sizes: list[int] = []
    snippets: list[str] = []

    for text in input_:
        # 对一个图对象需要embedding的文本按token分块
        split_texts = splitter.split_text(text)
        if split_texts is None:
            continue
        split_texts = [text for text in split_texts if len(text) > 0]

        sizes.append(len(split_texts))
        snippets.extend(split_texts)

    return snippets, sizes


def _reconstitute_embeddings(
        raw_embeddings: list[list[float]],
        sizes: list[int]
) -> list[list[float] | None]:
    """
    获取每一个原始文本的embedding
    :param raw_embeddings: 所有的embedding结果
    :param sizes: 原始文本的块数
    :return: 原始文本的embedding
    """
    embeddings: list[list[float] | None] = []
    cursor = 0
    for size in sizes:
        if size == 0:
            embeddings.append(None)
        elif size == 1:
            embedding = raw_embeddings[cursor]
            embeddings.append(embedding)
            cursor += 1
        else:
            # 多块去平均作为embedding
            chunk = raw_embeddings[cursor: cursor + size]
            average = np.average(chunk, axis=0)  # type: ignore
            normalized = average / np.linalg.norm(average)
            embeddings.append(normalized.tolist())
            cursor += size

    return embeddings
