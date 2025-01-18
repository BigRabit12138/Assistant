import logging

import tiktoken

DEFAULT_ENCODING_NAME = "cl100k_base"
log = logging.getLogger(__name__)


def num_tokens_from_string(
        string: str,
        model: str | None = None,
        encoding_name: str | None = None
) -> int:
    """
    计算文本token数量
    :param string: 输入文本
    :param model: 分词模型名称
    :param encoding_name: 分词模型名称
    :return:
    """
    if model is not None:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            msg = f"Failed to get encoding for {model} when getting \
            num_tokens_from_string. Fall back to default encoding {DEFAULT_ENCODING_NAME}"
            log.warning(msg)
            encoding = tiktoken.get_encoding(DEFAULT_ENCODING_NAME)
    else:
        encoding = tiktoken.get_encoding(encoding_name or DEFAULT_ENCODING_NAME)
    return len(encoding.encode(string))


def string_from_tokens(
        tokens: list[int],
        model: str | None = None,
        encoding_name: str | None = None
) -> str:
    if model is not None:
        encoding = tiktoken.encoding_for_model(model)
    elif encoding_name is not None:
        encoding = tiktoken.get_encoding(encoding_name)
    else:
        msg = "Either model or encoding_name must be specified."
        raise ValueError(msg)

    return encoding.decode(tokens)
