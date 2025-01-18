import networkx as nx

from datashaper import VerbCallbacks

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.llm import CompletionLLM
from assistant.memory.graphrag_v1.index.llm import load_llm
from assistant.memory.graphrag_v1.config.enums import LLMType
from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.graph.extractors.graph import GraphExtractor
from assistant.memory.graphrag_v1.index.verbs.entities.extraction.strategies.graph_intelligence.defaults import (
    DEFAULT_LLM_CONFIG
)
from assistant.memory.graphrag_v1.index.text_splitting import (
    TextSplitter,
    NoopTextSplitter,
    TokenTextSplitter,
)
from assistant.memory.graphrag_v1.index.verbs.entities.extraction.strategies.typing import (
    Document,
    EntityTypes,
    StrategyConfig,
    EntityExtractionResult,
)


async def run_gi(
        docs: list[Document],
        entity_types: EntityTypes,
        reporter: VerbCallbacks,
        pipeline_cache: PipelineCache,
        args: StrategyConfig,
) -> EntityExtractionResult:
    """
    对docs里面的文本进行实体抽取
    :param docs: 文本
    :param entity_types: 实体类型
    :param reporter: 进度条
    :param pipeline_cache: 缓存器
    :param args: 实体提取策略参数
    :return: 实体和实体图
    """
    # 获取模型配置
    llm_config = args.get("llm", DEFAULT_LLM_CONFIG)
    llm_type = llm_config.get("type", LLMType.StaticResponse)
    # 加载模型
    llm = load_llm("entity_extraction", llm_type, reporter, pipeline_cache, llm_config)
    # 提取实体
    return await run_extract_entities(llm, docs, entity_types, reporter, args)


async def run_extract_entities(
        llm: CompletionLLM,
        docs: list[Document],
        entity_types: EntityTypes,
        reporter: VerbCallbacks | None,
        args: StrategyConfig,
) -> EntityExtractionResult:
    """
    提取实体图
    :param llm: 大模型
    :param docs: 文本
    :param entity_types: 实体类型
    :param reporter: 显示回调器
    :param args: 运行配置参数
    :return: 实体抽取结果
    """
    # 加载参数
    encoding_name = args.get("encoding_name", "cl100k_base")

    prechunked = args.get("prechunked", False)
    chunk_size = args.get("chunk_size", defaults.CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", defaults.CHUNK_OVERLAP)

    tuple_delimiter = args.get("tuple_delimiter", None)
    record_delimiter = args.get("record_delimiter", None)
    completion_delimiter = args.get("completion_delimiter", None)
    extraction_prompt = args.get("extraction_prompt", None)
    encoding_model = args.get("encoding_name", None)
    max_gleanings = args.get("max_gleanings", defaults.ENTITY_EXTRACTION_MAX_GLEANINGS)

    # 获取文本分割器
    text_splitter = _create_text_splitter(
        prechunked,
        chunk_size,
        chunk_overlap,
        encoding_name
    )

    # 获取图对象提取器
    extractor = GraphExtractor(
        llm_invoker=llm,
        prompt=extraction_prompt,
        encoding_model=encoding_model,
        max_gleanings=max_gleanings,
        on_error=lambda e, s, d: (
            reporter.error("Entity Extraction Error", e, s, d) if reporter else None
        ),
    )
    text_list = [doc.text for doc in docs]

    if not prechunked:
        text_list = text_splitter.split_text("\n".join(text_list))

    # 提取
    results = await extractor(
        list(text_list),
        {
            "entity_types": entity_types,
            "tuple_delimiter": tuple_delimiter,
            "record_delimiter": record_delimiter,
            "completion_delimiter": completion_delimiter,
        },
    )

    graph = results.output
    for _, node in graph.nodes(data=True):
        if node is not None:
            # 转换ID
            node["source_id"] = ",".join(
                docs[int(id_)].id for id_ in node["source_id"].split(",")
            )

    for _, _, edge in graph.edges(data=True):
        if edge is not None:
            # 转换ID
            edge["source_id"] = ",".join(
                docs[int(id_)].id for id_ in edge["source_id"].split(",")
            )

    # 获取实体
    entities = [
        ({"name": item[0], **(item[1] or {})})
        for item in graph.nodes(data=True)
        if item is not None
    ]

    graph_data = "".join(nx.generate_graphml(graph))
    return EntityExtractionResult(entities, graph_data)


def _create_text_splitter(
        prechunked: bool,
        chunk_size: int,
        chunk_overlap: int,
        encoding_name: str
) -> TextSplitter:
    """
    创建文本分割器
    :param prechunked: 是否已经预先分割
    :param chunk_size: 块大小
    :param chunk_overlap: 块间重叠大小
    :param encoding_name: tokenizer模型
    :return: 文本分割器
    """
    if prechunked:
        return NoopTextSplitter()

    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
    )

