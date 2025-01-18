from datashaper import VerbCallbacks

from assistant.memory.graphrag_v1.llm import CompletionLLM
from assistant.memory.graphrag_v1.index.llm import load_llm

from assistant.memory.graphrag_v1.config.enums import LLMType
from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.graph.extractors.summarize import SummarizeExtractor
from assistant.memory.graphrag_v1.index.verbs.entities.summarize.strategies.typing import (
    StrategyConfig,
    SummarizedDescriptionResult,
)
from assistant.memory.graphrag_v1.index.verbs.entities.summarize.strategies.graph_intelligence.defaults import (
    DEFAULT_LLM_CONFIG,
)


async def run(
        described_items: str | tuple[str, str],
        descriptions: list[str],
        reporter: VerbCallbacks,
        pipeline_cache: PipelineCache,
        args: StrategyConfig,
) -> SummarizedDescriptionResult:
    """
    摘要一个图对象
    :param described_items: 图对象key
    :param descriptions: 图对象描述
    :param reporter: 进度条
    :param pipeline_cache: 大模型缓存器
    :param args: 摘要配置参数
    :return: 摘要结果
    """
    llm_config = args.get("llm", DEFAULT_LLM_CONFIG)
    llm_type = llm_config.get("type", LLMType.StaticResponse)
    # 加载大模型
    llm = load_llm(
        "summarize_descriptions", llm_type, reporter, pipeline_cache, llm_config
    )
    # 调用模型摘要
    return await run_summarize_descriptions(
        llm, described_items, descriptions, reporter, args
    )


async def run_summarize_descriptions(
        llm: CompletionLLM,
        items: str | tuple[str, str],
        descriptions: list[str],
        reporter: VerbCallbacks,
        args: StrategyConfig,
) -> SummarizedDescriptionResult:
    """
    调用模型摘要一个图对象
    :param llm: 大模型实例
    :param items: 图对象key
    :param descriptions: 图对象描述
    :param reporter: 进度条
    :param args: 摘要配置参数
    :return: 摘要结果
    """
    summarize_prompt = args.get("summarize_prompt", None)
    entity_name_key = args.get("entity_name_key", "entity_name")
    input_descriptions_key = args.get("input_descriptions_key", "description_list")
    max_tokens = args.get("max_tokens", None)

    # 加载摘要提取器
    extractor = SummarizeExtractor(
        llm_invoker=llm,
        summarization_prompt=summarize_prompt,
        entity_name_key=entity_name_key,
        input_descriptions_key=input_descriptions_key,
        on_error=lambda e, stack, details: (
            reporter.error("Entity Extraction Error", e, stack, details)
            if reporter
            else None
        ),
        max_summary_length=args.get("max_summary_length", None),
        max_input_tokens=max_tokens,
    )
    # 摘要
    result = await extractor(items=items, descriptions=descriptions)
    return SummarizedDescriptionResult(
        items=result.items,
        description=result.description,
    )
