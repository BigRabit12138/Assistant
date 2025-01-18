from openai import (
    AsyncOpenAI,
    AsyncAzureOpenAI,
)

# OpenAI客户端
OpenAIClientTypes = AsyncOpenAI | AsyncAzureOpenAI
