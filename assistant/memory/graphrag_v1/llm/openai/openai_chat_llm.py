import logging

from json import JSONDecodeError

from typing_extensions import Unpack

from assistant.memory.graphrag_v1.llm.base import BaseLLM
from assistant.memory.graphrag_v1.llm.types import (
    LLMInput,
    LLMOutput,
    CompletionInput,
    CompletionOutput,
)
from assistant.memory.graphrag_v1.llm.openai._json import clean_up_json
from assistant.memory.graphrag_v1.llm.openai.types import OpenAIClientTypes
from assistant.memory.graphrag_v1.llm.openai._prompts import JSON_CHECK_PROMPT
from assistant.memory.graphrag_v1.llm.openai.openai_configuration import OpenAIConfiguration
from assistant.memory.graphrag_v1.llm.openai.utils import (
    try_parse_json_object,
    get_completion_llm_args,
)

log = logging.getLogger(__name__)

_MAX_GENERATION_RETRIES = 3
FAILED_TO_CREATE_JSON_ERROR = "Failed to generate valid JSON output"


class OpenAIChatLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """
    OpenAI Chat模型
    """
    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(
            self,
            client: OpenAIClientTypes,
            configuration: OpenAIConfiguration
    ):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
            self,
            input_: CompletionInput,
            **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput | None:
        """
        调用OpenAI API
        :param input_: 模型输出
        :param kwargs: 模型参数
        :return: 模型返回结果
        """
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.configuration
        )
        history = kwargs.get("history") or []
        messages = [
            *history,
            {"role": "user", "content": input_},
        ]
        completion = await self.client.chat.completions.create(
            messages=messages,
            **args
        )
        return completion.choices[0].message.content

    async def _invoke_json(
            self,
            input_: CompletionInput,
            **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        """
        以支持JSON输出的方式调用大模型
        :param input_: 模型输入
        :param kwargs: 模型参数
        :return: 模型输出
        """
        name = kwargs.get("name") or "unknown"
        is_response_valid = kwargs.get("is_response_valid") or (lambda _x: True)

        async def generate(
                attempt: int | None = None,
        ) -> LLMOutput[CompletionOutput]:
            """
            以JSON输出的方式调用模型
            :param attempt: 重试次数
            :return: 模型输出结果
            """
            call_name = name if attempt is None else f"{name}@{attempt}"
            return (
                await self._native_json(input_, **{**kwargs, "name": call_name})
                if self.configuration.model_supports_json
                else await self._manual_json(input_, **{**kwargs, "name": call_name})
            )

        def is_valid(x: dict | None) -> bool:
            """
            判断模型相应是否是JSON格式
            :param x: 模型输出文本
            :return: 是否符合格式
            """
            return x is not None and is_response_valid(x)

        result = await generate()
        retry = 0
        # 如果不符合格式，重试
        while not is_valid(result.json) and retry < _MAX_GENERATION_RETRIES:
            result = await generate(retry)
            retry += 1

        if is_valid(result.json):
            return result
        raise RuntimeError(FAILED_TO_CREATE_JSON_ERROR)

    async def _native_json(
            self,
            input_: CompletionInput,
            **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        """
        以原生支持JSON输出的方式调用模型
        :param input_: 模型输入
        :param kwargs: 模型参数
        :return: 模型输出
        """
        result = await self._invoke(
            input_,
            **{
                **kwargs,
                "model_parameter": {
                    **(kwargs.get("model_parameters") or {}),
                    "response_format": {"type": "json_object"},
                },
            },
        )

        raw_output = result.output or ""
        # 反序列化输出
        raw_output = clean_up_json(raw_output)
        json_output = try_parse_json_object(raw_output)

        return LLMOutput[CompletionOutput](
            output=raw_output,
            json=json_output,
            history=result.history
        )

    async def _manual_json(
            self,
            input_: CompletionInput,
            **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        """
        以手动支持JSON输出的方式调用模型
        :param input_: 模型输入
        :param kwargs: 模型参数
        :return: 模型输出
        """
        result = await self._invoke(input_, **kwargs)
        history = result.history or []
        # 过滤JSON序列化文本
        output = clean_up_json(result.output or "")

        try:
            json_output = try_parse_json_object(output)
            return LLMOutput[CompletionOutput](
                output=output, json=json_output, history=history
            )
        except (TypeError, JSONDecodeError):
            log.warning("error paring llm json, retrying.")
            result = await self._try_clean_json_with_llm(output, **kwargs)
            output = clean_up_json(result.output or "")
            json = try_parse_json_object(output)

            return LLMOutput[CompletionOutput](
                output=output,
                json=json,
                history=history,
            )

    async def _try_clean_json_with_llm(
            self,
            output: str,
            **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        """
        调用大模型矫正JSON格式
        :param output: 模型输出
        :param kwargs: 模型参数
        :return: 模型输出
        """
        name = kwargs.get("name") or "unknown"
        return await self._invoke(
            JSON_CHECK_PROMPT,
            **{
                **kwargs,
                "variables": {"input_text": output},
                "name": f"fix_json@{name}",
            },
        )
