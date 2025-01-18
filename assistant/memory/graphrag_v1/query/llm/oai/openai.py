import logging

from typing import Any

from tenacity import (
    Retrying,
    RetryError,
    AsyncRetrying,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential_jitter,
)

from assistant.memory.graphrag_v1.query.llm.base import BaseLLMCallback
from assistant.memory.graphrag_v1.query.llm.oai.base import OpenAILLMImpl
from assistant.memory.graphrag_v1.query.llm.oai.typing import (
    OpenaiApiType,
    OPENAI_RETRY_ERRORS_TYPES,
)

log = logging.getLogger(__name__)


class OpenAI(OpenAILLMImpl):
    def __init__(
            self,
            api_key: str,
            model: str,
            deployment_name: str | None = None,
            api_base: str | None = None,
            api_version: str | None = None,
            api_type: OpenaiApiType = OpenaiApiType.OpenAI,
            organization: str | None = None,
            max_retries: int = 10,
            retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERRORS_TYPES,
    ):
        OpenAILLMImpl.__init__(
            self=self,
            api_key=api_key,
            deployment_name=deployment_name,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,
            organization=organization,
            max_retries=max_retries,
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
            log.exception(f"RetryError at generate(): {str(e)}")
            return ""
        else:
            return ""

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
            log.exception(f"Error at agenerate(): {str(e)}")
            return ""
        else:
            return ""

    def _generate(
            self,
            messages: str | list[Any],
            streaming: bool = True,
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> str:
        response = self.sync_client.chat.completions.create(
            model=self.model,
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

    async def _agenerate(
            self,
            messages: str | list[Any],
            streaming: bool = True,
            callbacks: list[BaseLLMCallback] | None = None,
            **kwargs: Any,
    ) -> str:
        response = await self.async_client.chat.completions.create(
            model=self.model,
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
