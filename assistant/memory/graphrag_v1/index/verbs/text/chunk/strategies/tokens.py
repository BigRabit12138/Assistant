from typing import Any
from collections.abc import Iterable

import tiktoken

from datashaper import ProgressTicker

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.index.text_splitting import Tokenizer
from assistant.memory.graphrag_v1.index.verbs.text.chunk.typing import TextChunk


def run(
        input_: list[str],
        args: dict[str, Any],
        tick: ProgressTicker
) -> Iterable[TextChunk]:
    """
    对输入的文本按token数量切分
    :param input_: 输入文本
    :param args: 切分策略参数
    :param tick: 进度条
    :return: 分好块的文本
    """
    tokens_per_chunk = args.get("chunk_size", defaults.CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", defaults.CHUNK_OVERLAP)
    encoding_name = args.get("encoding_name", defaults.ENCODING_MODEL)
    enc = tiktoken.get_encoding(encoding_name)

    def encode(text: str) -> list[int]:
        """
        编码
        :param text: 文本
        :return: token
        """
        if not isinstance(text, str):
            text = f"{text}"
        return enc.encode(text)

    def decode(tokens: list[int]) -> str:
        """
        解码
        :param tokens: token
        :return: 文本
        """
        return enc.decode(tokens)

    return split_text_on_tokens(
        input_,
        Tokenizer(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
            encode=encode,
            decode=decode,
        ),
        tick,
    )


def split_text_on_tokens(
        texts: list[str],
        enc: Tokenizer,
        tick: ProgressTicker
) -> list[TextChunk]:
    """
    对文本按token分块
    :param texts: 文本
    :param enc: 分词器
    :param tick: 进度条
    :return: 分块的文本
    """
    result = []
    mapped_ids = []

    # 编码
    for source_doc_idx, text in enumerate(texts):
        encoded = enc.encode(text)
        tick(1)
        mapped_ids.append((source_doc_idx, encoded))

    # 切分成单个token
    input_ids: list[tuple[int, int]] = [
        (source_doc_idx, id_) for source_doc_idx, ids in mapped_ids for id_ in ids
    ]

    # 获取第一块
    start_idx = 0
    cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx: cur_idx]
    while start_idx < len(input_ids):
        # token块解码成文本
        chunk_text = enc.decode([id_ for _, id_ in chunk_ids])
        # 获取token块的文档编号
        doc_indices = list({doc_idx for doc_idx, _ in chunk_ids})

        result.append(
            TextChunk(
                text_chunk=chunk_text,
                source_doc_indices=doc_indices,
                n_tokens=len(chunk_ids),
            )
        )

        # 获取下一块
        start_idx += enc.tokens_per_chunk - enc.chunk_overlap
        cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx: cur_idx]

    return result
