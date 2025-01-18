import asyncio
import logging

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential_jitter,
)
from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.limiting import LLMLimiter
from assistant.memory.graphrag_v1.llm.errors import RetriesExhaustedError
from assistant.memory.graphrag_v1.llm.types import (
    LLM,
    LLMInput,
    LLMConfig,
    LLMOutput,
    LLMInvocationFn,
    LLMInvocationResult,
)

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")
TRateLimiterError = TypeVar("TRateLimiterError", bound=BaseException)

_CANNOT_MEASURE_INPUT_TOKENS_MSG = "cannot measure input tokens"
_CANNOT_MEASURE_OUTPUT_TOKENS_MSG = "cannot measure output tokens"

log = logging.getLogger(__name__)


class RateLimitingLLM(LLM[TIn, TOut], Generic[TIn, TOut]):
    """
    限制速率的大模型
    """
    _delegate: LLM[TIn, TOut]
    _rate_limiter: LLMLimiter | None
    _semaphore: asyncio.Semaphore | None
    _count_tokens: Callable[[str], int]
    _config: LLMConfig
    _operation: str
    _retryable_errors: list[type[Exception]]
    _rate_limit_errors: list[type[Exception]]
    _on_invoke: LLMInvocationFn
    _extract_sleep_recommendation: Callable[[Any], float]

    def __init__(
            self,
            delegate: LLM[TIn, TOut],
            config: LLMConfig,
            operation: str,
            retryable_errors: list[type[Exception]],
            rate_limit_errors: list[type[Exception]],
            rate_limiter: LLMLimiter | None = None,
            semaphore: asyncio.Semaphore | None = None,
            count_tokens: Callable[[str], int] | None = None,
            get_sleep_time: Callable[[BaseException], float] | None = None,
    ):
        self._delegate = delegate
        self._rate_limiter = rate_limiter
        self._semaphore = semaphore
        self._config = config
        self._operation = operation
        self._retryable_errors = retryable_errors
        self._rate_limit_errors = rate_limit_errors
        self._count_tokens = count_tokens or (lambda _s: -1)
        self._extract_sleep_recommendation = get_sleep_time or (lambda _e: 0.0)
        self._on_invoke = lambda _v: None

    def on_invoke(self, fn: LLMInvocationFn | None) -> None:
        """
        设置invoke回调函数
        :param fn: 回调函数
        :return:
        """
        self._on_invoke = fn or (lambda _v: None)

    def count_request_tokens(self, input_: TIn) -> int:
        """
        计算文本的Token数量
        :param input_: 输入文本
        :return: Token数量
        """
        if isinstance(input_, str):
            return self._count_tokens(input_)

        if isinstance(input_, list):
            result = 0
            for item in input_:
                if isinstance(item, str):
                    result += self._count_tokens(item)
                elif isinstance(item, dict):
                    result += self._count_tokens(item.get("content", ""))
                else:
                    raise TypeError(_CANNOT_MEASURE_INPUT_TOKENS_MSG)
            return result
        raise TypeError(_CANNOT_MEASURE_INPUT_TOKENS_MSG)

    def count_response_tokens(self, output: TOut | None) -> int:
        """
        计算输出结果的文本token数
        :param output: 模型输出
        :return: token数
        """
        if output is None:
            return 0
        if isinstance(output, str):
            return self._count_tokens(output)
        if isinstance(output, list) and all(isinstance(x, str) for x in output):
            return sum(self._count_tokens(item) for item in output)
        if isinstance(output, list):
            return 0
        raise TypeError(_CANNOT_MEASURE_OUTPUT_TOKENS_MSG)

    async def __call__(
            self,
            input_: TIn,
            **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[TOut]:
        """
        以带重试的方式调用大模型
        :param input_: 输出文本
        :param kwargs: 模型参数
        :return: 模型输出
        """
        name = kwargs.get("name", "Process")
        attempt_number = 0
        call_times: list[float] = []
        input_tokens = self.count_request_tokens(input_)

        max_retries = self._config.max_retries or 10
        max_retry_wait = self._config.max_retry_wait or 10
        follow_recommendation = self._config.sleep_on_rate_limit_recommendation
        retryer = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(max=max_retry_wait),
            reraise=True,
            retry=retry_if_exception_type(tuple(self._retryable_errors)),
        )

        async def sleep_for(time: float | None) -> None:
            """
            休眠等待
            :param time: 休眠等待时间
            :return:
            """
            log.warning(
                f"{name} failed to invoke LLM {attempt_number}/{max_retries} attempts. Cause: \
                rate limit exceeded, will retry. Recommended sleep for {time} seconds. \
                Follow recommendation? {follow_recommendation}"
            )
            if follow_recommendation and time:
                await asyncio.sleep(time)

            raise

        async def do_attempt() -> LLMOutput[TOut]:
            """
            尝试调用大模型
            :return: 模型输出结果
            """
            nonlocal call_times
            call_start = asyncio.get_event_loop().time()

            try:
                # 调用大模型
                return await self._delegate(input_, **kwargs)
            except BaseException as e:
                if isinstance(e, tuple(self._rate_limit_errors)):
                    # 计算休眠时间
                    sleep_time = self._extract_sleep_recommendation(e)
                    # 休眠
                    await sleep_for(sleep_time)
                raise
            finally:
                # 计算调用耗时
                call_end = asyncio.get_event_loop().time()
                call_times.append(call_end - call_start)

        async def execute_with_retry() -> tuple[LLMOutput[TOut], float]:
            """
            以带重试的方式调用大模型
            :return: 模型输出结果和成功调用的开始时间
            """
            nonlocal attempt_number
            async for attempt in retryer:
                with attempt:
                    if self._rate_limiter and input_tokens > 0:
                        # 等待满足限制要求
                        await self._rate_limiter.acquire(input_tokens)
                    start_ = asyncio.get_event_loop().time()
                    attempt_number += 1
                    return await do_attempt(), start_

            log.error(f"Retries exhausted for {name}.")
            raise RetriesExhaustedError(name, max_retries)

        result: LLMOutput[TOut]
        start = 0.0

        # 尝试调用模型获取结果
        if self._semaphore is None:
            result, start = await execute_with_retry()
        else:
            async with self._semaphore:
                result, start = await execute_with_retry()

        end = asyncio.get_event_loop().time()
        output_tokens = self.count_response_tokens(result.output)
        if self._rate_limiter and output_tokens > 0:
            # 等待满足限制要求
            await self._rate_limiter.acquire(output_tokens)

        invocation_result = LLMInvocationResult(
            result=result,
            name=name,
            num_retries=attempt_number - 1,
            total_time=end - start,
            call_times=call_times,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self._handle_invoke_result(invocation_result)
        return result

    def _handle_invoke_result(
            self,
            result: LLMInvocationResult[LLMOutput[TOut]]
    ) -> None:
        """
        执行模型成功调用回调
        :param result: 模型输出结果
        :return:
        """
        log.info(
            f'perf - llm.{self._operation} "{result.name}" with {result.num_retries} retries \
            took {result.total_time}. input_tokens={result.input_tokens}, output_tokens={result.output_tokens}.'
        )
        self._on_invoke(result)
