from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.base import BaseLLM
from assistant.memory.graphrag_v1.llm.base.base_llm import TIn, TOut
from assistant.memory.graphrag_v1.llm.openai.types import OpenAIClientTypes
from assistant.memory.graphrag_v1.llm.openai.openai_configuration import OpenAIConfiguration
from assistant.memory.graphrag_v1.llm.types import (
    LLMInput,
    LLMOutput,
    EmbeddingInput,
    EmbeddingOutput,
)


class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """
    openai embedding模型
    """
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
            input_: EmbeddingInput,
            **kwargs: Unpack[LLMInput],
    ) -> EmbeddingOutput | None:
        """
        调用openai embedding模型获取embedding
        :param input_: 输入文本
        :param kwargs: 模型参数
        :return: embedding
        """
        args = {
            "model": self.configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        # openai embedding模型api调用
        embedding = await self.client.embeddings.create(
            input=input_,
            **args,
        )
        return [d.embedding for d in embedding.data]

    async def _invoke_json(
            self,
            input_: TIn,
            **kwargs: Unpack[LLMInput]
    ) -> LLMOutput:
        """
        没用
        :param input_:
        :param kwargs:
        :return:
        """
        pass
