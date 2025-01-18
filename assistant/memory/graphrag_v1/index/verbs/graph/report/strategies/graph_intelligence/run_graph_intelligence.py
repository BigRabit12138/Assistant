import json
import logging
import traceback

from datashaper import VerbCallbacks

from assistant.memory.graphrag_v1.llm import CompletionLLM
from assistant.memory.graphrag_v1.index.llm import load_llm
from assistant.memory.graphrag_v1.config.enums import LLMType
from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.utils.rate_limiter import RateLimiter
from assistant.memory.graphrag_v1.index.verbs.graph.report.strategies.graph_intelligence.defaults import (
    MOCK_RESPONSES,
)
from assistant.memory.graphrag_v1.index.graph.extractors.community_reports import (
    CommunityReportsExtractor,
)
from assistant.memory.graphrag_v1.index.verbs.graph.report.strategies.typing import (
    StrategyConfig,
    CommunityReport,
)

log = logging.getLogger(__name__)


async def run(
        community: str | int,
        input_: str,
        level: int,
        reporter: VerbCallbacks,
        pipeline_cache: PipelineCache,
        args: StrategyConfig,
) -> CommunityReport | None:
    """
    调用大语言模型生成聚簇摘要
    :param community: 聚簇id
    :param input_: 聚簇的文本
    :param level: 聚簇的层次
    :param reporter: 回调钩子
    :param pipeline_cache: 缓存器
    :param args: 大模型参数
    :return: 摘要结果
    """
    llm_config = args.get(
        "llm", {"type": LLMType.StaticResponse, "responses": MOCK_RESPONSES}
    )
    llm_type = llm_config.get("type", LLMType.StaticResponse)
    # 加载大模型
    llm = load_llm(
        "community_reporting", llm_type, reporter, pipeline_cache, llm_config
    )
    # 运行大模型摘要
    return await _run_extractor(llm, community, input_, level, args, reporter)


async def _run_extractor(
        llm: CompletionLLM,
        community: str | int,
        input_: str,
        level: int,
        args: StrategyConfig,
        reporter: VerbCallbacks,
) -> CommunityReport | None:
    """
    调用大语言模型生成聚簇摘要
    :param llm: 大模型
    :param community: 聚簇id
    :param input_: 聚簇的文本
    :param level: 聚簇的层次
    :param args: 大模型参数
    :param reporter: 回调钩子
    :return: 摘要结果
    """
    rate_limiter = RateLimiter(rate=1, per=60)
    extractor = CommunityReportsExtractor(
        llm,
        extraction_prompt=args.get("extraction_prompt", None),
        max_report_length=args.get("max_report_length", None),
        on_error=lambda e_, stack, _data: reporter.error(
            "Community Report Extraction Error", e_, stack
        ),
    )

    try:
        # 限制速率
        await rate_limiter.acquire()
        # 获取摘要结果
        results = await extractor({"input_text": input_})
        report = results.structured_output
        if report is None or len(report.keys()) == 0:
            log.warning(f"No report found for community: {community}.")
            return None

        return CommunityReport(
            community=community,
            full_content=results.output,
            level=level,
            rank=_parse_rank(report),
            title=report.get("title", f"Community Report: {community}"),
            rank_explanation=report.get("rating_explanation", ""),
            summary=report.get("summary", ""),
            findings=report.get("findings", []),
            full_content_json=json.dumps(report, indent=4),
        )
    except Exception as e:
        log.exception(f"Error processing community: {community}.")
        reporter.error("Community Report Extraction Error", e, traceback.format_exc())
        return None


def _parse_rank(report: dict) -> float:
    """
    获取rating
    :param report: 解析结果
    :return: rating
    """
    rank = report.get("rating", -1)
    try:
        return float(rank)
    except ValueError:
        log.exception(f"Error parsing rank: {rank} defaulting to -1.")
        return -1
