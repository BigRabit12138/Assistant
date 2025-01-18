from itertools import islice
from collections.abc import Iterator

import tiktoken


def num_tokens(
        text: str,
        token_encoder: tiktoken.Encoding | None = None
) -> int:
    """
    计算文本的token数量
    :param text: 文本
    :param token_encoder: 分词器
    :return: 文本的token数量
    """
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    return len(token_encoder.encode(text))


def batched(iterable: Iterator, n: int):
    if n < 1:
        value_error = "n must be at least one"
        raise ValueError(value_error)

    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def chunk_text(
        text: str,
        max_tokens: int,
        token_encoder: tiktoken.Encoding | None = None
):
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    tokens = token_encoder.encode(text)
    chunk_iterator = batched(iter(tokens), max_tokens)
    yield from chunk_iterator
