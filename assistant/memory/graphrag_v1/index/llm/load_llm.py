from __future__ import annotations

import asyncio
import logging

from typing import TYPE_CHECKING, Any

from assistant.memory.graphrag_v1.config.enums import LLMType
from assistant.memory.graphrag_v1.llm import (
    LLMCache,
    LLMLimiter,
    EmbeddingLLM,
    CompletionLLM,
    MockCompletionLLM,
    OpenAIConfiguration,
    create_openai_client,
    create_openai_chat_llm,
    create_tpm_rpm_limiters,
    create_openai_embedding_llm,
    create_openai_completion_llm,
)

if TYPE_CHECKING:
    from datashaper import VerbCallbacks

    from assistant.memory.graphrag_v1.index.cache import PipelineCache
    from assistant.memory.graphrag_v1.index.typing import ErrorHandlerFn

log = logging.getLogger(__name__)

# 模型并发限制器
_semaphores: dict[str, asyncio.Semaphore] = {}
# 模型速率限制器
_rate_limiters: dict[str, LLMLimiter] = {}


def load_llm(
        name: str,
        llm_type: LLMType,
        callbacks: VerbCallbacks,
        cache: PipelineCache | None,
        llm_config: dict[str, Any] | None = None,
        chat_only=False,
) -> CompletionLLM:
    """
    加载模型
    :param name: 模型用途
    :param llm_type: 模型类型
    :param callbacks: 回调函数
    :param cache: 缓存器
    :param llm_config: 模型参数配置
    :param chat_only: 是否为Chat模型
    :return: 大模型
    """
    on_error = _create_error_handler(callbacks)

    if llm_type in loaders:
        if chat_only and not loaders[llm_type]["chat"]:
            msg = f"LLM type {llm_type} does not support chat."
            raise ValueError(msg)
        if cache is not None:
            cache = cache.child(name)

        loader = loaders[llm_type]
        return loader["load"](on_error, cache, llm_config or {})

    msg = f"Unknown LLM type {llm_type}."
    raise ValueError(msg)


def load_llm_embeddings(
        name: str,
        llm_type: LLMType,
        callbacks: VerbCallbacks,
        cache: PipelineCache | None,
        llm_config: dict[str, Any] | None = None,
        chat_only=False,
) -> EmbeddingLLM:
    """
    加载embedding模型
    :param name: 本次embedding名称
    :param llm_type: 模型类型
    :param callbacks: 回调钩子
    :param cache: 缓存器
    :param llm_config: 模型配置
    :param chat_only: 是否只要chat模型
    :return: embedding模型
    """
    # 添加异常回调
    on_error = _create_error_handler(callbacks)
    if llm_type in loaders:
        if chat_only and not loaders[llm_type]["chat"]:
            msg = f"LLM type {llm_type} does not support chat."
            raise ValueError(msg)
        if cache is not None:
            cache = cache.child(name)

        return loaders[llm_type]["load"](on_error, cache, llm_config or {})

    msg = f"Unknown LLM type {llm_type}."
    raise ValueError(msg)


def _create_error_handler(callbacks: VerbCallbacks) -> ErrorHandlerFn:
    """
    Error处理器
    :param callbacks: 回调函数
    :return: Error处理函数
    """
    def on_error(
            error: BaseException | None = None,
            stack: str | None = None,
            details: dict | None = None,
    ) -> None:
        callbacks.error("Error Invoking LLM", error, stack, details)

    return on_error


def _load_openai_completion_llm(
        on_error: ErrorHandlerFn,
        cache: LLMCache,
        config: dict[str, Any],
        azure=False,
):
    return _create_openai_completion_llm(
        OpenAIConfiguration(
            {
                **_get_base_config(config),
                "model": config.get("model", "gpt-4-turbo-preview"),
                "deployment_name": config.get("deployment_name"),
                "temperature": config.get("temperature", 0.0),
                "frequency_penalty": config.get("frequency_penalty", 0),
                "presence_penalty": config.get("presence_penalty", 0),
                "top_p": config.get("top_p", 1),
                "max_tokens": config.get("max_tokens", 4000),
                "n": config.get("n"),
            }
        ),
        on_error,
        cache,
        azure,
    )


def _load_openai_chat_llm(
        on_error: ErrorHandlerFn,
        cache: LLMCache,
        config: dict[str, Any],
        azure=False,
):
    """
    加载OpenAI Chat模型
    :param on_error: 错误处理器
    :param cache: 模型缓存器
    :param config: 模型配置
    :param azure: 是否使用微软Azure
    :return: 大模型
    """
    return _create_openai_chat_llm(
        OpenAIConfiguration(
            {
                **_get_base_config(config),
                "model": config.get("model", "gpt-4-turbo-preview"),
                "deployment_name": config.get("deployment_name"),
                "temperature": config.get("temperature", 0.0),
                "frequency_penalty": config.get("frequency_penalty", 0),
                "presence_penalty": config.get("presence_penalty", 0),
                "top_p": config.get("top_p", 1),
                "max_tokens": config.get("max_tokens"),
                "n": config.get("n"),
            }
        ),
        on_error,
        cache,
        azure,
    )


def _load_openai_embeddings_llm(
        on_error: ErrorHandlerFn,
        cache: LLMCache,
        config: dict[str, Any],
        azure=False,
) -> EmbeddingLLM:
    """
    加载openai embedding模型
    :param on_error: 错误回调
    :param cache: 缓存器
    :param config: 模型配置
    :param azure: 是否使用azure
    :return: embedding模型
    """
    return _create_openai_embeddings_llm(
        OpenAIConfiguration(
            {
                **_get_base_config(config),
                "model": config.get(
                    "embeddings_model", config.get("model", "text-embedding-3-small")
                ),
                "deployment_name": config.get("deployment_name"),
            }
        ),
        on_error,
        cache,
        azure,
    )


def _load_azure_openai_completion_llm(
        on_error: ErrorHandlerFn,
        cache: LLMCache,
        config: dict[str, Any]
):
    return _load_openai_completion_llm(on_error, cache, config, azure=True)


def _load_azure_openai_chat_llm(
        on_error: ErrorHandlerFn,
        cache: LLMCache,
        config: dict[str, Any]
):
    return _load_openai_chat_llm(on_error, cache, config, azure=True)


def _load_azure_openai_embeddings_llm(
        on_error: ErrorHandlerFn,
        cache: LLMCache,
        config: dict[str, Any]
):
    return _load_openai_embeddings_llm(on_error, cache, config, azure=True)


def _get_base_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    获取加载模型所必须的参数
    :param config: 配置
    :return: 配置
    """
    api_key = config.get("api_key")

    return {
        **config,
        "api_key": api_key,
        "api_base": config.get("api_base"),
        "api_version": config.get("api_version"),
        "organization": config.get("organization"),
        "proxy": config.get("proxy"),
        "max_retries": config.get("max_retires", 10),
        "request_timeout": config.get("request_timeout", 60.0),
        "model_support_json": config.get("model_support_json"),
        "concurrent_requests": config.get("concurrent_requests", 4),
        "encoding_model": config.get("encoding_model", "cl100k_base"),
        "cognitive_services_endpoint": config.get("cognitive_services_endpoint"),
    }


def _load_static_response(
        _on_error: ErrorHandlerFn,
        _cache: PipelineCache,
        config: dict[str, Any],
) -> CompletionLLM:
    return MockCompletionLLM(config.get("response", []))


# 模型加载器
loaders = {
    LLMType.OpenAI: {
        "load": _load_openai_completion_llm,
        "chat": False,
    },
    LLMType.AzureOpenAI: {
        "load": _load_azure_openai_completion_llm,
        "chat": False,
    },
    LLMType.OpenAIChat: {
        "load": _load_openai_chat_llm,
        "chat": True,
    },
    LLMType.AzureOpenAIChat: {
        "load": _load_azure_openai_chat_llm,
        "chat": True,
    },
    LLMType.OpenAIEmbedding: {
        "load": _load_openai_embeddings_llm,
        "chat": False,
    },
    LLMType.AzureOpenAIEmbedding: {
        "load": _load_azure_openai_embeddings_llm,
        "chat": False,
    },
    LLMType.StaticResponse: {
        "load": _load_static_response,
        "chat": False,
    },
}


def _create_openai_chat_llm(
        configuration: OpenAIConfiguration,
        on_error: ErrorHandlerFn,
        cache: LLMCache,
        azure=False,
) -> CompletionLLM:
    """
    加载OpenAI Chat模型
    :param configuration: 模型配置
    :param on_error: 错误处理函数
    :param cache: 缓存器
    :param azure: 是否使用Azure
    :return: 大模型
    """
    client = create_openai_client(configuration=configuration, azure=azure)
    limiter = _create_limiter(configuration)
    semaphore = _create_semaphore(configuration)
    return create_openai_chat_llm(
        client, configuration, cache, limiter, semaphore, on_error=on_error
    )


def _create_openai_completion_llm(
        configuration: OpenAIConfiguration,
        on_error: ErrorHandlerFn,
        cache: LLMCache,
        azure=False,
) -> CompletionLLM:
    client = create_openai_client(configuration=configuration, azure=azure)
    limiter = _create_limiter(configuration)
    semaphore = _create_semaphore(configuration)
    return create_openai_completion_llm(
        client, configuration, cache, limiter, semaphore, on_error=on_error
    )


def _create_openai_embeddings_llm(
        configuration: OpenAIConfiguration,
        on_error: ErrorHandlerFn,
        cache: LLMCache,
        azure=False,
) -> EmbeddingLLM:
    """
    加载openai embedding模型
    :param configuration: 模型配置
    :param on_error: 错误回调
    :param cache: 缓存器
    :param azure: 是否使用azure
    :return: embedding模型
    """
    # 创建模型实例
    client = create_openai_client(configuration=configuration, azure=azure)
    # 创建速率限制器
    limiter = _create_limiter(configuration)
    # 创建并发控制器
    semaphore = _create_semaphore(configuration)
    return create_openai_embedding_llm(
        client, configuration, cache, limiter, semaphore, on_error=on_error
    )


def _create_limiter(configuration: OpenAIConfiguration) -> LLMLimiter:
    """
    获取模型对用的速率限制器
    :param configuration: 模型配置
    :return: 模型限制器
    """
    limit_name = configuration.model or configuration.deployment_name or "default"
    if limit_name not in _rate_limiters:
        tpm = configuration.tokens_per_minute
        rpm = configuration.requests_per_minute
        log.info(f"create TPM/RPM limiter for {limit_name}: TPM={tpm}, RPM={rpm}.")
        _rate_limiters[limit_name] = create_tpm_rpm_limiters(configuration)

    return _rate_limiters[limit_name]


def _create_semaphore(configuration: OpenAIConfiguration) -> asyncio.Semaphore | None:
    """
    获取并发控制器
    :param configuration: 配置
    :return: 并发控制器
    """
    limit_name = configuration.model or configuration.deployment_name or "default"
    concurrency = configuration.concurrent_requests

    if not concurrency:
        log.info(f"no concurrency limiter for {limit_name}.")
        return None

    if limit_name not in _semaphores:
        log.info(f"create concurrency limiter for {limit_name}: {concurrency}.")
        _semaphores[limit_name] = asyncio.Semaphore(concurrency)

    return _semaphores[limit_name]
