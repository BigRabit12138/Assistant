from typing import Union, Any
from abc import ABC, abstractmethod
from collections.abc import Callable

from openai import (
    OpenAI,
    AzureOpenAI,
    AsyncOpenAI,
    AsyncAzureOpenAI,
)

from assistant.memory.graphrag_v1.query.llm.base import BaseTextEmbedding
from assistant.memory.graphrag_v1.query.llm.oai.typing import OpenaiApiType
from assistant.memory.graphrag_v1.query.progress import (
    StatusReporter,
    ConsoleStatusReporter,
)


class BaseOpenAILLM(ABC):
    _async_client: AsyncOpenAI | AsyncAzureOpenAI
    _sync_client: OpenAI | AzureOpenAI

    def __init__(self):
        self._create_openai_client()

    @abstractmethod
    def _create_openai_client(self):
        pass

    def set_clients(
            self,
            sync_client: OpenAI | AzureOpenAI,
            async_client: AsyncOpenAI | AsyncAzureOpenAI,
    ):
        self._sync_client = sync_client
        self._async_client = async_client

    @property
    def async_client(self) -> AsyncOpenAI | AsyncAzureOpenAI | None:
        return self._async_client

    @property
    def sync_client(self) -> OpenAI | AzureOpenAI | None:
        return self._sync_client

    @async_client.setter
    def async_client(self, client: AsyncOpenAI | AsyncAzureOpenAI):
        self._async_client = client

    @sync_client.setter
    def sync_client(self, client: OpenAI | AzureOpenAI):
        self._sync_client = client


class OpenAILLMImpl(BaseOpenAILLM):
    _reporter: StatusReporter = ConsoleStatusReporter()

    def __init__(
            self,
            api_key: str | None = None,
            azure_ad_token_provider: Union[Callable, None] = None,
            deployment_name: str | None = None,
            api_base: str | None = None,
            api_version: str | None = None,
            api_type: OpenaiApiType = OpenaiApiType.OpenAI,
            organization: str | None = None,
            max_retries: int = 10,
            request_timeout: float = 180.0,
            reporter: StatusReporter | None = None,
    ):
        self.api_key = api_key
        self.azure_ad_token_provider = azure_ad_token_provider
        self.deployment_name = deployment_name
        self.api_base = api_base
        self.api_version = api_version
        self.api_type = api_type
        self.organization = organization
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.reporter = reporter or ConsoleStatusReporter()

        try:
            super().__init__()
        except Exception as e:
            self._reporter.error(
                message="Failed to create OpenAI client",
                details={self.__class__.__name__: str(e)},
            )
            raise

    def _create_openai_client(self):
        if self.api_type == OpenaiApiType.AzureOpenAI:
            if self.api_base is None:
                msg = "api_base is required for Azure OpenAI"
                raise ValueError(msg)

            sync_client = AzureOpenAI(
                api_key=self.api_key,
                azure_ad_token_provider=self.azure_ad_token_provider,
                organization=self.organization,
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                azure_deployment=self.deployment_name,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            async_client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_ad_token_provider=self.azure_ad_token_provider,
                organization=self.organization,
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                azure_deployment=self.deployment_name,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
            self.set_clients(sync_client=sync_client, async_client=async_client)

        else:
            sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                organization=self.organization,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                organization=self.organization,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            self.set_clients(sync_client=sync_client, async_client=async_client)


class OpenAITextEmbeddingImpl(BaseTextEmbedding):
    _reporter: StatusReporter | None = None

    def _create_openai_client(self, api_type: OpenaiApiType):
        pass

    def embed(
            self,
            text: str,
            **kwargs: Any,
    ) -> list[float]:
        pass

    async def aembed(
            self,
            text: str,
            **kwargs: Any,
    ) -> list[float]:
        pass
