import json
import logging

from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Literal, cast
from collections.abc import (
    Callable,
    Collection,
    Iterable
)

import tiktoken
import pandas as pd

from assistant.memory.graphrag_v1.index.utils import (
    num_tokens_from_string
)

EncodedText = list[int]
DecodeFn = Callable[[EncodedText], str]
EncodeFn = Callable[[str], EncodedText]
LengthFn = Callable[[str], int]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tokenizer:
    """
    分词器
    """
    chunk_overlap: int
    tokens_per_chunk: int
    decode: DecodeFn
    encode: EncodeFn


class TextSplitter(ABC):
    """
    文本分割基类
    """
    _chunk_size: int
    _chunk_overlap: int
    _length_function: LengthFn
    _keep_separator: bool
    _add_start_index: bool
    _strip_whitespace: bool

    def __init__(
            self,
            chunk_size: int = 8191,
            chunk_overlap: int = 100,
            length_function: LengthFn = len,
            keep_separator: bool = False,
            add_start_index: bool = False,
            strip_whitespace: bool = True
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(
            self,
            text: str | list[str]
    ) -> Iterable[str]:
        pass


class NoopTextSplitter(TextSplitter):
    """
    空文本分割器
    """
    def split_text(
            self,
            text: str | list[str]
    ) -> Iterable[str]:
        return [text] if isinstance(text, str) else text


class TokenTextSplitter(TextSplitter):
    """
    按Tokens数量分割文本
    """
    _allowed_special: Literal["all"] | set[str]
    _disallowed_special: Literal["all"] | Collection[str]

    def __init__(
            self,
            encoding_name: str = "cl100k_base",
            model_name: str | None = None,
            allowed_special: Literal["all"] | set[str] | None = None,
            disallowed_special: Literal["all"] | Collection[str] = "all",
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        if model_name is not None:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                log.exception(f"Model {model_name} not found, using {encoding_name}")
                enc = tiktoken.get_encoding(encoding_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        if allowed_special is None:
            allowed_special = set()
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special

    def encode(self, text: str) -> list[int]:
        """
        文本编码成token
        :param text: 文本
        :return: token
        """
        return self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special
        )

    def num_tokens(self, text: str) -> int:
        """
        计算文本token数量
        :param text: 文本
        :return: 文本token数量
        """
        return len(self.encode(text))

    def split_text(
            self,
            text: str | list[str]
    ) -> Iterable[str]:
        """
        将文本按token数量分开
        :param text: 文本
        :return: 分块的文本
        """
        if cast(bool, pd.isna(text)) or text == "":
            return []
        if isinstance(text, list):
            text = " ".join(text)
        if not isinstance(text, str):
            msg = f"Attempting to split a non-string value, actual is {type(text)}."
            raise TypeError(msg)
        # 获取分词器
        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=lambda text_: self.encode(text_)
        )
        # 开切
        return split_text_on_tokens(
            text=text,
            tokenizer=tokenizer
        )


class TextListSplitterType(str, Enum):
    DELIMITED_STRING = "delimited_string"
    JSON = "json"


class TextListSplitter(TextSplitter):
    def __init__(
            self,
            chunk_size: int,
            splitter_type: TextListSplitterType = TextListSplitterType.JSON,
            input_delimiter: str | None = None,
            output_delimiter: str | None = None,
            model_name: str | None = None,
            encoding_name: str | None = None
    ):
        super().__init__(chunk_size, chunk_overlap=0)
        self._type = splitter_type
        self._input_delimiter = input_delimiter
        self._output_delimiter = output_delimiter or "\n"
        self._length_function = lambda x: num_tokens_from_string(
            x, model=model_name, encoding_name=encoding_name
        )

    def split_text(
            self,
            text: str | list[str]
    ) -> Iterable[str]:
        if not text:
            return []

        result: list[str] = []
        current_chunk: list[str] = []

        current_length: int = self._length_function("[]")
        string_list = self._load_text_list(text)

        if len(string_list) == 1:
            return string_list

        for item in string_list:
            item_length = self._length_function(f"{item},")

            if current_length + item_length > self._chunk_size:
                if current_chunk and len(current_chunk) > 0:
                    self._append_to_result(
                        result, current_chunk
                    )
                    current_chunk = [item]
                    current_length = item_length
            else:
                current_chunk.append(item)
                current_length += item_length
        self._append_to_result(result, current_chunk)

        return result

    def _load_text_list(self, text: str | list[str]):
        if isinstance(text, list):
            string_list = text
        elif self._type == TextListSplitterType.JSON:
            string_list = json.loads(text)
        else:
            string_list = text.split(self._input_delimiter)
        return string_list

    def _append_to_result(
            self,
            chunk_list: list[str],
            new_chunk: list[str]
    ):
        if new_chunk and len(new_chunk) > 0:
            if self._type == TextListSplitterType.JSON:
                chunk_list.append(json.dumps(new_chunk))
            else:
                chunk_list.append(self._output_delimiter.join(new_chunk))


def split_text_on_tokens(
        *,
        text: str,
        tokenizer: Tokenizer
) -> list[str]:
    """
    将文本按token分块
    :param text: 文本
    :param tokenizer: 分词器
    :return: 分块文本
    """
    splits: list[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    # 获取第一块
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx: cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        # 获取下一块
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx: cur_idx]
    return splits
