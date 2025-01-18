import asyncio

from typing import Any, Union
from collections.abc import Callable

import numpy as np
import tiktoken

from tenacity import (
    Retrying,
    RetryError,
    AsyncRetrying,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential_jitter,
)

from assistant.memory.graphrag_v1.query.llm.oai.base import OpenAILLMImpl
from assistant.memory.graphrag_v1.query.llm.base import BaseTextEmbedding
from assistant.memory.graphrag_v1.query.llm.oai.typing import (
    OpenaiApiType,
    OPENAI_RETRY_ERRORS_TYPES,
)

from assistant.memory.graphrag_v1.query.llm.text_utils import chunk_text
from assistant.memory.graphrag_v1.query.progress import StatusReporter


class OpenAIEmbedding(BaseTextEmbedding, OpenAILLMImpl):
    def __init__(
            self,
            api_key: str | None = None,
            model: str = "text-embedding-3-small",
            azure_ad_token_provider: Union[Callable, None] = None,
            deployment_name: str | None = None,
            api_base: str | None = None,
            api_version: str | None = None,
            api_type: OpenaiApiType = OpenaiApiType.OpenAI,
            organization: str | None = None,
            encoding_name: str = "cl100k_base",
            max_tokens: int = 8191,
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
        self.encoding_name = encoding_name
        self.max_tokens = max_tokens
        self.token_encoder = tiktoken.get_encoding(self.encoding_name)
        self.retry_error_types = retry_error_types

    def embed(
            self,
            text: str,
            **kwargs: Any,
    ) -> list[float]:
        token_chunks = chunk_text(
            text=text,
            token_encoder=self.token_encoder,
            max_tokens=self.max_tokens,
        )
        chunk_embeddings = []
        chunk_lens = []
        for chunk in token_chunks:
            try:
                embedding, chunk_len = self._embed_with_retry(chunk, **kwargs)
                chunk_embeddings.append(embedding)
                chunk_lens.append(chunk_len)
            except Exception as e:
                self._reporter.error(
                    message="Error embedding chunk",
                    details={self.__class__.__name__: str(e)},
                )

                continue
        chunk_embeddings = np.average(
            np.array(chunk_embeddings),
            axis=0,
            weights=chunk_lens
        )
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        return chunk_embeddings.tolist()

    async def aembed(
            self,
            text: str,
            **kwargs: Any,
    ) -> list[float]:
        token_chunks = chunk_text(
            text=text,
            token_encoder=self.token_encoder,
            max_tokens=self.max_tokens,
        )
        chunk_embeddings = []
        chunk_lens = []

        embedding_results = await asyncio.gather(*[
            self._aembed_with_retry(chunk, **kwargs) for chunk in token_chunks
        ])
        embedding_results = [result for result in embedding_results if result[0]]
        chunk_embeddings = [result[0] for result in embedding_results]
        chunk_lens = [result[1] for result in embedding_results]

        chunk_embeddings = np.average(
            np.array(chunk_embeddings),
            axis=0,
            weights=chunk_lens
        )
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        return chunk_embeddings.tolist()

    def _embed_with_retry(
            self,
            text: str | tuple,
            **kwargs: Any,
    ) -> tuple[list[float], int]:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    embedding = (
                        self.sync_client.embeddings.create(
                            input=text,
                            model=self.model,
                            **kwargs,
                        )
                        .data[0]
                        .embedding
                        or []
                    )
                    return embedding, len(text)
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return [], 0
        else:
            return [], 0

    async def _aembed_with_retry(
            self,
            text: str | tuple,
            **kwargs: Any,
    ) -> tuple[list[float], int]:
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            async for attempt in retryer:
                with attempt:
                    embedding = (
                        await self.async_client.embeddings.create(
                            input=text,
                            model=self.model,
                            **kwargs,
                        )
                    ).data[0].embedding or []
                    return embedding, len(text)
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return [], 0
        else:
            return [], 0
