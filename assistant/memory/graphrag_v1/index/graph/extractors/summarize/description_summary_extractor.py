import json

from dataclasses import dataclass

from assistant.memory.graphrag_v1.llm import CompletionLLM
from assistant.memory.graphrag_v1.index.typing import ErrorHandlerFn
from assistant.memory.graphrag_v1.index.utils.tokens import num_tokens_from_string
from assistant.memory.graphrag_v1.index.graph.extractors.summarize.prompts import SUMMARIZE_PROMPT

DEFAULT_MAX_INPUT_TOKENS = 4_000
DEFAULT_MAX_SUMMARY_LENGTH = 500


@dataclass
class SummarizationResult:
    """
    摘要结果
    """
    items: str | tuple[str, str]
    description: str
    

class SummarizeExtractor:
    """
    摘要提取器
    """
    _llm: CompletionLLM
    _entity_name_key: str
    _input_descriptions_key: str
    _summarization_prompt: str
    _on_error: ErrorHandlerFn
    _max_summary_length: int
    _max_input_tokens: int
    
    def __init__(
            self,
            llm_invoker: CompletionLLM,
            entity_name_key: str | None = None,
            input_descriptions_key: str | None = None,
            summarization_prompt: str | None = None,
            on_error: ErrorHandlerFn | None = None,
            max_summary_length: int | None = None,
            max_input_tokens: int | None = None,
    ):
        self._llm = llm_invoker
        self._entity_name_key = entity_name_key or "entity_name"
        self._input_descriptions_key = input_descriptions_key or "description_list"

        self._summarization_prompt = summarization_prompt or SUMMARIZE_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_summary_length = max_summary_length or DEFAULT_MAX_SUMMARY_LENGTH
        self._max_input_tokens = max_input_tokens or DEFAULT_MAX_INPUT_TOKENS

    async def __call__(
            self,
            items: str | tuple[str, str],
            descriptions: list[str]
    ) -> SummarizationResult:
        """
        调用模型摘要文本
        :param items: 图对象key
        :param descriptions: 图对象描述文本
        :return: 摘要结果
        """
        result = ""
        if len(descriptions) == 0:
            result = ''
        if len(descriptions) == 1:
            result = descriptions[0]
        else:
            result = await self._summarize_descriptions(
                items, descriptions
            )

        return SummarizationResult(
            items=items,
            description=result or "",
        )

    async def _summarize_descriptions(
           self,
            items: str | tuple[str, str],
            descriptions: list[str]
    ) -> str:
        """
        对文本调用大模型获取摘要，如果无法一次完成，递归摘要
        :param items: 图对象key
        :param descriptions: 图对象描述
        :return: 摘要
        """
        sorted_items = sorted(items) if isinstance(items, list) else items

        if not isinstance(descriptions, list):
            descriptions = [descriptions]
        # 获取剩余可用token数量
        usable_tokens = self._max_input_tokens - num_tokens_from_string(
            self._summarization_prompt
        )
        descriptions_collected = []
        result = ""

        for i, description in enumerate(descriptions):
            usable_tokens -= num_tokens_from_string(description)  # type: ignore
            descriptions_collected.append(description)

            # 如果无法全部一次性完成摘要，递归进行
            if (usable_tokens < 0 and len(descriptions_collected) > 1) or (
                i == len(descriptions) - 1
            ):
                result = await self._summarize_descriptions_with_llm(
                    sorted_items, descriptions_collected
                )
                # 还未完成，递归
                if i != len(descriptions) - 1:
                    descriptions_collected = [result]
                    # 更新可用token
                    usable_tokens = (
                        self._max_input_tokens
                        - num_tokens_from_string(self._summarization_prompt)
                        - num_tokens_from_string(result)
                    )

        return result

    async def _summarize_descriptions_with_llm(
            self,
            items: str | tuple[str, str] | list[str],
            descriptions: list[str]
    ) -> str:
        """
        调用大模型对摘要文本
        :param items: 图像对象key
        :param descriptions: 图对象描述
        :return: 大模型原始输出的摘要
        """
        response = await self._llm(
            self._summarization_prompt,
            name="summarize",
            variables={
                self._entity_name_key: json.dumps(items),
                self._input_descriptions_key: json.dumps(sorted(descriptions)),
            },
            model_parameters={"max_tokens": self._max_summary_length},
        )
        return str(response.output)
