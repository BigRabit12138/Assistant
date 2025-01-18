import logging

from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.base import BaseLLM
from assistant.memory.graphrag_v1.llm.base.base_llm import TIn, TOut
from assistant.memory.graphrag_v1.llm.openai.types import OpenAIClientTypes
from assistant.memory.graphrag_v1.llm.openai.utils import get_completion_llm_args
from assistant.memory.graphrag_v1.llm.openai.openai_configuration import OpenAIConfiguration
from assistant.memory.graphrag_v1.llm.types import (
    LLMInput,
    CompletionInput,
    CompletionOutput,
)

log = logging.getLogger(__name__)


class OpenAICompletionLLM(BaseLLM[CompletionInput, CompletionOutput]):
    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(
            self,
            client: OpenAIClientTypes,
            configuration: OpenAIConfiguration,
    ):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
            self,
            input_: CompletionInput,
            **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput | None:
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.configuration
        )
        completion = self.client.completions.create(prompt=input_, **args)
        return completion.choices[0].text
