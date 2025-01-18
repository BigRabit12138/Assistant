from typing import Any, Union
from collections.abc import (
    Callable,
    Generator,
    AsyncGenerator,
)

from tenacity import (
    Retrying,
    RetryError,
    AsyncRetrying,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential_jitter,
)

from assistant.memory.graphrag_v1.query.progress import StatusReporter
from assistant.memory.graphrag_v1.query.llm.oai.base import OpenAILLMImpl
from assistant.memory.graphrag_v1.query.llm.base import (
    BaseLLM,
    BaseLLMCallback,
)
from assistant.memory.graphrag_v1.query.llm.oai.typing import (
    OpenaiApiType,
    OPENAI_RETRY_ERRORS_TYPES,
)

_MODEL_REQUIRED_MSG = "model is required"


class ChatOpenAI(BaseLLM, OpenAILLMImpl):
    def __init__(
            self,
            api_key: str | None = None,
            model: str | None = None,
            azure_ad_token_provider: Union[Callable, None] = None,
            deployment_name: str | None = None,
            api_base: str | None = None,
            api_version: str | None = None,
            api_type: OpenaiApiType = OpenaiApiType.OpenAI,
            organization: str | None = None,
            max_retries: int = 10,
            request_timeout: float = 180.0,
            retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERRORS_TYPES,
            reporter: StatusReporter | None = None,
    ):
        OpenAILLMImpl.__init__(
            self=self,
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            deployment_name=deployment_name,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,
            organization=organization,
            max_retries=max_retries,
            request_timeout=request_timeout,
            reporter=reporter,
        )
        self.model = model
        self.retry_error_types = retry_error_types

    def generate(
            self,
            messages: str | list[Any],
            streaming: bool = True,
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> str:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    return self._generate(
                        messages=messages,
                        streaming=streaming,
                        callbacks=callbacks,
                        **kwargs,
                    )
        except RetryError as e:
            self._reporter.error(
                message="Error at generate()", details={self.__class__.__name__: str(e)}
            )
            return ""
        else:
            return ""

    def stream_generate(
            self,
            messages: str | list[Any],
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> Generator[str, None, None]:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    generator = self._stream_generate(
                        messages=messages,
                        callbacks=callbacks,
                        **kwargs,
                    )
                    yield from generator
        except RetryError as e:
            self._reporter.error(
                message="Error at stream_generate()",
                details={self.__class__.__name__: str(e)},
            )
            return
        else:
            return

    async def agenerate(
            self,
            messages: str | list[Any],
            streaming: bool = True,
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> str:
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            async for attempt in retryer:
                with attempt:
                    return await self._agenerate(
                        messages=messages,
                        streaming=streaming,
                        callbacks=callbacks,
                        **kwargs,
                    )
        except RetryError as e:
            self._reporter.error(
                message=f"Error at agenerate(): {e}"
            )
            return ""
        else:
            return ""

    async def astream_generate(
            self,
            messages: str | list[Any],
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            async for attempt in retryer:
                with attempt:
                    generator = self._astream_generate(
                        messages=messages,
                        callbacks=callbacks,
                        **kwargs,
                    )
                    async for response in generator:
                        yield response
        except RetryError as e:
            self._reporter.error(f"Error at astream_generate(): {e}")
            return
        else:
            return

    def _generate(
            self,
            messages: str | list[Any],
            streaming: bool = True,
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> str:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)
        response = self.sync_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=streaming,
            **kwargs,
        )
        if streaming:
            full_response = ""
            while True:
                try:
                    chunk = response.__next__()
                    if not chunk or not chunk.choices:
                        continue

                    delta = (
                        chunk.choices[0].delta.content
                        if chunk.choices[0].delta and chunk.choices[0].delta.content
                        else ""
                    )

                    full_response += delta
                    if callbacks:
                        for callback in callbacks:
                            callback.on_llm_new_token(delta)
                    if chunk.choices[0].finish_reason == "stop":
                        break
                except StopIteration:
                    break
            return full_response
        return response.choices[0].message.content or ""

    def _stream_generate(
            self,
            messages: str | list[Any],
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> Generator[str, None, None]:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)
        response = self.sync_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        for chunk in response:
            if not chunk or not chunk.choices:
                continue

            delta = (
                chunk.choices[0].delta.content
                if chunk.choices[0].delta and chunk.choices[0].delta.content
                else ""
            )
            yield delta

            if callbacks:
                for callback in callbacks:
                    callback.on_llm_new_token(delta)

    async def _agenerate(
            self,
            messages: str | list[Any],
            streaming: bool = True,
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> str:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=streaming,
            **kwargs,
        )
        if streaming:
            full_response = ""
            while True:
                try:
                    chunk = await response.__anext__()
                    if not chunk or not chunk.choices:
                        continue

                    delta = (
                        chunk.choices[0].delta.content
                        if chunk.choices[0].delta and chunk.choices[0].delta.content
                        else ""
                    )

                    full_response += delta
                    if callbacks:
                        for callback in callbacks:
                            callback.on_llm_new_token(delta)
                    if chunk.choices[0].finish_reason == "stop":
                        break
                except StopIteration:
                    break
            return full_response
        return response.choices[0].message.content or ""

    async def _astream_generate(
            self,
            messages: str | list[Any],
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)

        response = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        async for chunk in response:
            if not chunk or not chunk.choices:
                continue

            delta = (
                chunk.choices[0].delta.content
                if chunk.choices[0].delta and chunk.choices[0].delta.content
                else ""
            )

            yield delta

            if callbacks:
                for callback in callbacks:
                    callback.on_llm_new_token(delta)
