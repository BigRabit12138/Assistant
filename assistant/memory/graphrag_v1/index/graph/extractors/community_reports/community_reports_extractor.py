import logging
import traceback

from typing import Any
from dataclasses import dataclass

from assistant.memory.graphrag_v1.llm import CompletionLLM
from assistant.memory.graphrag_v1.index.typing import ErrorHandlerFn
from assistant.memory.graphrag_v1.index.utils import dict_has_keys_with_types
from assistant.memory.graphrag_v1.index.graph.extractors.community_reports.prompts import COMMUNITY_REPORT_PROMPT

log = logging.getLogger(__name__)


@dataclass
class CommunityReportsResult:
    """
    摘要结果
    """
    output: str
    structured_output: dict


class CommunityReportsExtractor:
    """
    聚簇信息摘要器
    """
    _llm: CompletionLLM
    _input_text_key: str
    _extraction_prompt: str
    _output_formatter_prompt: str
    _on_error: ErrorHandlerFn
    _max_report_length: int

    def __init__(
            self,
            llm_invoker: CompletionLLM,
            input_text_key: str | None = None,
            extraction_prompt: str | None = None,
            on_error: ErrorHandlerFn | None = None,
            max_report_length: int | None = None,
    ):
        self._llm = llm_invoker
        self._input_text_key = input_text_key or "input_text"
        self._extraction_prompt = extraction_prompt or COMMUNITY_REPORT_PROMPT
        self._on_error = on_error or (lambda _e, _s, _d: None)
        self._max_report_length = max_report_length or 1500

    async def __call__(self, inputs: dict[str, Any]):
        """
        调用大模型对文本进行摘要，并解析结果
        :param inputs: 输入文本
        :return: 摘要结果
        """
        output = None
        try:
            # 调用大模型获取结果
            response = (
                await self._llm(
                    self._extraction_prompt,
                    json=True,
                    name="create_community_report",
                    variables={self._input_text_key: inputs[self._input_text_key]},
                    is_response_valid=lambda x: dict_has_keys_with_types(
                        x,
                        [
                            ("title", str),
                            ("summary", str),
                            ("findings", list),
                            ("rating", float),
                            ("rating_explanation", str)
                        ],
                    ),
                    model_parameters={"max_tokens": self._max_report_length},
                )
                or {}
            )
            output = response.json or {}
        except Exception as e:
            log.exception("error generating community report.")
            self._on_error(e, traceback.format_exc(), None)
            output = {}
        # 解析结果
        text_output = self._get_text_output(output)
        return CommunityReportsResult(
            structured_output=output,
            output=text_output,
        )

    @staticmethod
    def _get_text_output(parsed_output: dict) -> str:
        """
        解析所有的摘要结果
        :param parsed_output: 大模型输出摘要结果
        :return: 解析的所有的摘要文本
        """
        title = parsed_output.get("title", "Report")
        summary = parsed_output.get("summary", "")
        findings = parsed_output.get("findings", [])

        def finding_summary(finding: dict):
            """
            获取摘要
            :param finding: 大模型输出的摘要
            :return: 摘要文本
            """
            if isinstance(finding, str):
                return finding
            return finding.get("summary")

        def finding_explanation(finding: dict):
            """
            获取解释
            :param finding: 大模型输出摘要的解释
            :return: 解释文本
            """
            if isinstance(finding, str):
                return ""
            return finding.get("explanation")

        report_sections = "\n\n".join(
            f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
        )
        return f"# {title}\n\n{summary}\n\n{report_sections}"
