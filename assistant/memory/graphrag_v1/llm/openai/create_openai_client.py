import logging

from functools import cache

from openai import AsyncOpenAI, AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from assistant.memory.graphrag_v1.llm.openai.types import OpenAIClientTypes
from assistant.memory.graphrag_v1.llm.openai.openai_configuration import OpenAIConfiguration

log = logging.getLogger(__name__)

API_BASE_REQUIRED_FOR_AZURE = "api_base is required for Azure OpenAI client"


@cache
def create_openai_client(
        configuration: OpenAIConfiguration,
        azure: bool
) -> OpenAIClientTypes:
    """
    创建OpenAI模型实例
    :param configuration: 模型配置
    :param azure: 是否使用Azure
    :return: OpenAI模型实例
    """
    if azure:
        api_base = configuration.api_base
        if api_base is None:
            raise ValueError(API_BASE_REQUIRED_FOR_AZURE)

        log.info(
            f"Creating Azure OpenAI client api_base={api_base}, deployment_name={configuration.deployment_name}."
        )
        if configuration.cognitive_service_endpoint is None:
            cognitive_services_endpoint = "https://cognitiveservices.azure.com/.default"
        else:
            cognitive_services_endpoint = configuration.cognitive_service_endpoint

        return AsyncAzureOpenAI(
            api_key=configuration.api_key if configuration.api_key else None,
            azure_ad_token_provider=get_bearer_token_provider(
                DefaultAzureCredential(), cognitive_services_endpoint
            )
            if not configuration.api_key
            else None,
            organization=configuration.organization,
            api_version=configuration.api_version,
            azure_endpoint=api_base,
            azure_deployment=configuration.deployment_name,
            timeout=configuration.request_timeout or 180.0,
            max_retries=0,
        )

    log.info(f"Creating OpenAI client base_url={configuration.api_base}")
    return AsyncOpenAI(
        api_key=configuration.api_key,
        base_url=configuration.api_base,
        organization=configuration.organization,
        timeout=configuration.request_timeout or 180.0,
        max_retries=0,
    )
