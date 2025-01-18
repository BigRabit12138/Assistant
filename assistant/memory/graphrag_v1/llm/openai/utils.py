import json
import logging

from typing import Any
from collections.abc import Callable

import tiktoken
from openai import (
    RateLimitError,
    APIConnectionError,
    InternalServerError,
)

from assistant.memory.graphrag_v1.llm.openai.openai_configuration import OpenAIConfiguration

# 默认分词器名称
DEFAULT_ENCODING = "cl100k_base"

# 分词器
_encoders: dict[str, tiktoken.Encoding] = {}

# 默认重试异常
RETRYABLE_ERRORS: list[type[Exception]] = [
    RateLimitError,
    APIConnectionError,
    InternalServerError,
]
# 默认速率限制异常
RATE_LIMIT_ERRORS: list[type[Exception]] = [RateLimitError]

log = logging.getLogger(__name__)


def get_token_counter(
        config: OpenAIConfiguration
) -> Callable[[str], int]:
    """
    获取Tokens长度计算器
    :param config: 模型配置
    :return: 长度计算器
    """
    model = config.encoding_model or "cl100k_base"
    enc = _encoders.get(model)
    if enc is None:
        enc = tiktoken.get_encoding(model)
        _encoders[model] = enc

    return lambda s: len(enc.encode(s))


def perform_variable_replacements(
        input_: str,
        history: list[dict],
        variables: dict | None
) -> str:
    """
    实例化模板内容
    :param input_: 模板
    :param history: 历史记录
    :param variables: 变量
    :return: 实例化的模板
    """
    result = input_

    def replace_all(input__: str) -> str:
        result_ = input__
        if variables:
            for entry_ in variables:
                result_ = result_.replace(f"{{{entry_}}}", variables[entry_])
        return result_

    result = replace_all(result)
    # 实例化大模型输出
    for i in range(len(history)):
        entry = history[i]
        if entry.get("role") == "system":
            history[i]["content"] = replace_all(entry.get("content") or "")

    return result


def get_completion_cache_args(
        configuration: OpenAIConfiguration
) -> dict:
    """
    获取模型参数
    :param configuration: 模型配置
    :return: 配置参数
    """
    return {
        "model": configuration.model,
        "temperature": configuration.temperature,
        "frequency_penalty": configuration.frequency_penalty,
        "presence_penalty": configuration.presence_penalty,
        "top_p": configuration.top_p,
        "max_tokens": configuration.max_tokens,
        "n": configuration.n
    }


def get_completion_llm_args(
        parameters: dict | None,
        configuration: OpenAIConfiguration
) -> dict:
    """
    获取模型参数
    :param parameters: 模型参数
    :param configuration: 模型配置参数
    :return:
    """
    return {
        **get_completion_cache_args(configuration),
        **(parameters or {}),
    }


def try_parse_json_object(input_: str) -> dict:
    """
    反序列化JSON字符串
    :param input_: JSON序列化字符串
    :return: 反序列化JSON字符串
    """
    try:
        result = json.loads(input_)
    except json.JSONDecodeError:
        log.exception(f"error loading json, json={input_}")
        raise
    else:
        if not isinstance(result, dict):
            raise TypeError
        return result


def get_sleep_time_from_error(e: Any) -> float:
    """
    计算遇到可以重试的异常的休眠时间
    :param e: 异常
    :return: 休眠时间
    """
    sleep_time = 0.0
    if isinstance(e, RateLimitError) and _please_retry_after in str(e):
        sleep_time = int(str(e).split(_please_retry_after)[1].split(" second")[0])

    return sleep_time


_please_retry_after = "Please retry after "
