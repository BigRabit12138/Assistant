import asyncio

from assistant.memory.graphrag_v1.llm.limiting import LLMLimiter
from assistant.memory.graphrag_v1.llm.openai.types import OpenAIClientTypes
from assistant.memory.graphrag_v1.llm.base import CacheingLLM, RateLimitingLLM
from assistant.memory.graphrag_v1.llm.openai.openai_chat_llm import OpenAIChatLLM
from assistant.memory.graphrag_v1.llm.openai.json_parsing_llm import JsonParsingLLM
from assistant.memory.graphrag_v1.llm.openai.openai_configuration import OpenAIConfiguration
from assistant.memory.graphrag_v1.llm.openai.openai_completion_llm import OpenAICompletionLLM
from assistant.memory.graphrag_v1.llm.openai.openai_embeddings_llm import OpenAIEmbeddingsLLM
from assistant.memory.graphrag_v1.llm.openai.openai_token_replacing_llm import OpenAITokenReplacingLLM
from assistant.memory.graphrag_v1.llm.openai.openai_history_tracking_llm import OpenAIHistoryTrackingLLM
from assistant.memory.graphrag_v1.llm.openai.utils import (
    RETRYABLE_ERRORS,
    RATE_LIMIT_ERRORS,
    get_token_counter,
    get_sleep_time_from_error,
    get_completion_cache_args,
)
from assistant.memory.graphrag_v1.llm.types import (
    LLM,
    LLMCache,
    EmbeddingLLM,
    CompletionLLM,
    ErrorHandlerFn,
    LLMInvocationFn,
    OnCacheActionFn,
)


def create_openai_chat_llm(
        client: OpenAIClientTypes,
        config: OpenAIConfiguration,
        cache: LLMCache | None = None,
        limiter: LLMLimiter | None = None,
        semaphore: asyncio.Semaphore | None = None,
        on_invoke: LLMInvocationFn | None = None,
        on_error: ErrorHandlerFn | None = None,
        on_cache_hit: OnCacheActionFn | None = None,
        on_cache_miss: OnCacheActionFn | None = None,
) -> CompletionLLM:
    """
    给模型装配功能
    :param client: OpenAI模型实例
    :param config: 模型配置
    :param cache: 缓存器
    :param limiter: 速率限制器
    :param semaphore: 并发控制器
    :param on_invoke: 模型启动回调函数
    :param on_error: 模型错误回调函数
    :param on_cache_hit: 找到缓存回调函数
    :param on_cache_miss: 缺失缓存回调函数
    :return: 大模型
    """
    operation = "chat"
    result = OpenAIChatLLM(client, config)
    result.on_error(on_error)
    if limiter is not None or semaphore is not None:
        result = _rate_limited(
            result,
            config,
            operation,
            limiter,
            semaphore,
            on_invoke
        )
    if cache is not None:
        result = _cached(
            result,
            config,
            operation,
            cache,
            on_cache_hit,
            on_cache_miss
        )
    result = OpenAIHistoryTrackingLLM(result)
    result = OpenAITokenReplacingLLM(result)
    return JsonParsingLLM(result)


def create_openai_completion_llm(
        client: OpenAIClientTypes,
        config: OpenAIConfiguration,
        cache: LLMCache | None = None,
        limiter: LLMLimiter | None = None,
        semaphore: asyncio.Semaphore | None = None,
        on_invoke: LLMInvocationFn | None = None,
        on_error: ErrorHandlerFn | None = None,
        on_cache_hit: OnCacheActionFn | None = None,
        on_cache_miss: OnCacheActionFn | None = None,
) -> CompletionLLM:
    operation = "completion"
    result = OpenAICompletionLLM(client, config)
    result.on_error(on_error)
    if limiter is not None or semaphore is not None:
        result = _rate_limited(
            result,
            config,
            operation,
            limiter,
            semaphore,
            on_invoke
        )
    if cache is not None:
        result = _cached(
            result,
            config,
            operation,
            cache,
            on_cache_hit,
            on_cache_miss
        )
    return OpenAITokenReplacingLLM(result)


def create_openai_embedding_llm(
        client: OpenAIClientTypes,
        config: OpenAIConfiguration,
        cache: LLMCache | None = None,
        limiter: LLMLimiter | None = None,
        semaphore: asyncio.Semaphore | None = None,
        on_invoke: LLMInvocationFn | None = None,
        on_error: ErrorHandlerFn | None = None,
        on_cache_hit: OnCacheActionFn | None = None,
        on_cache_miss: OnCacheActionFn | None = None,
) -> EmbeddingLLM:
    """
    创建openai embedding模型
    :param client: 模型实例
    :param config: 模型配置
    :param cache: 缓存器
    :param limiter: 速率限制器
    :param semaphore: 并发控制器
    :param on_invoke: 开始回调
    :param on_error: 错误回调
    :param on_cache_hit: 存在缓存回调
    :param on_cache_miss: 不存在缓存回调
    :return: embedding模型
    """
    operation = "embedding"
    result = OpenAIEmbeddingsLLM(client, config)
    result.on_error(on_error)
    if limiter is not None or semaphore is not None:
        result = _rate_limited(
            result,
            config,
            operation,
            limiter,
            semaphore,
            on_invoke
        )
    if cache is not None:
        result = _cached(
            result,
            config,
            operation,
            cache,
            on_cache_hit,
            on_cache_miss
        )

    return result


def _rate_limited(
        delegate: LLM,
        config: OpenAIConfiguration,
        operation: str,
        limiter: LLMLimiter | None,
        semaphore: asyncio.Semaphore | None,
        on_invoke: LLMInvocationFn | None,
):
    """
    给模型装配速率限制
    :param delegate: 模型
    :param config: 模型配置
    :param operation: 模型类型
    :param limiter: 速率限制器
    :param semaphore: 并发限制器
    :param on_invoke: 模型回调
    :return: 大模型
    """
    result = RateLimitingLLM(
        delegate,
        config,
        operation,
        RETRYABLE_ERRORS,
        RATE_LIMIT_ERRORS,
        limiter,
        semaphore,
        get_token_counter(config),
        get_sleep_time_from_error,
    )
    result.on_invoke(on_invoke)
    return result


def _cached(
        delegate: LLM,
        config: OpenAIConfiguration,
        operation: str,
        cache: LLMCache,
        on_cache_hit: OnCacheActionFn | None,
        on_cache_miss: OnCacheActionFn | None,
):
    """
    给模型装配缓存器
    :param delegate: 模型
    :param config: 配置
    :param operation: 模型操作类型
    :param cache: 缓存器
    :param on_cache_hit: 找到缓存回调
    :param on_cache_miss: 缺失缓存回调
    :return: 大模型
    """
    cache_args = get_completion_cache_args(config)
    result = CacheingLLM(delegate, cache_args, operation, cache)
    result.on_cache_hit(on_cache_hit)
    result.on_cache_miss(on_cache_miss)
    return result
