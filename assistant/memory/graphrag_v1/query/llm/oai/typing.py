from enum import Enum
from typing import Any, cast

import openai


OPENAI_RETRY_ERRORS_TYPES = (
    cast(Any, openai).RateLimitError,
    cast(Any, openai).APIConnectionError,
)


class OpenaiApiType(str, Enum):
    OpenAI = "openai"
    AzureOpenAI = "azure"
