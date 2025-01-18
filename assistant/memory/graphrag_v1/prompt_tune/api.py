from datashaper import NoopVerbCallbacks
from pydantic import PositiveInt, validate_call

from assistant.memory.graphrag_v1.index.llm import load_llm
from assistant.memory.graphrag_v1.prompt_tune.types import DocSelectionType
from assistant.memory.graphrag_v1.index.progress import PrintProgressReporter
from assistant.memory.graphrag_v1.config.models.graph_rag_config import GraphRagConfig
from assistant.memory.graphrag_v1.prompt_tune.loader import (
    MIN_CHUNK_SIZE,
    load_docs_in_chunks,
)
from assistant.memory.graphrag_v1.prompt_tune.generator import (
    MAX_TOKEN_COUNT,
    detect_language,
    generate_domain,
    generate_persona,
    generate_entity_types,
    create_entity_extraction_prompt,
    generate_community_reporter_role,
    generate_community_report_rating,
    create_entity_summarization_prompt,
    generate_entity_relationship_examples,
    create_community_summarization_prompt,
)


@validate_call
async def generate_indexing_prompts(
        config: GraphRagConfig,
        root: str,
        chunk_size: PositiveInt = MIN_CHUNK_SIZE,
        limit: PositiveInt = 15,
        selection_method: DocSelectionType = DocSelectionType.RANDOM,
        domain: str | None = None,
        language: str | None = None,
        max_tokens: int = MAX_TOKEN_COUNT,
        skip_entity_types: bool = False,
        min_examples_required: PositiveInt = 2,
        n_subset_max: PositiveInt = 300,
        k: PositiveInt = 15,
) -> tuple[str, str, str]:
    reporter = PrintProgressReporter("")
    doc_list = await load_docs_in_chunks(
        root=root,
        config=config,
        limit=limit,
        select_method=str(selection_method),
        reporter=reporter,
        chunk_size=chunk_size,
        n_subset_max=n_subset_max,
        k=k,
    )

    llm = load_llm(
        "prompt_tuning",
        config.llm.type,
        NoopVerbCallbacks(),
        None,
        config.llm.model_dump(),
    )

    if not domain:
        reporter.info("Generating domain...")
        domain = await generate_domain(llm, doc_list)
        reporter.info(f"Generated domain: {domain}")

    if not language:
        reporter.info("Detecting language...")
        language = await detect_language(llm, doc_list)

    reporter.info("Generating persona...")
    persona = await generate_persona(llm, domain)

    reporter.info("Generating community report ranking description...")
    community_report_ranking = await generate_community_report_rating(
        llm, domain=domain, persona=persona, docs=doc_list
    )

    entity_types = None
    if not skip_entity_types:
        reporter.info("Generating entity types...")
        entity_types = await generate_entity_types(
            llm,
            domain=domain,
            persona=persona,
            docs=doc_list,
            json_mode=config.llm.model_supports_json or False,
        )

    reporter.info("Generating entity relationship examples...")
    examples = await generate_entity_relationship_examples(
        llm,
        persona=persona,
        entity_types=entity_types,
        docs=doc_list,
        language=language,
        json_mode=False,
    )

    reporter.info("Generating entity extraction prompt...")
    entity_extraction_prompt = create_entity_extraction_prompt(
        entity_types=entity_types,
        docs=doc_list,
        examples=examples,
        language=language,
        json_mode=False,
        encoding_model=config.encoding_model,
        max_token_count=max_tokens,
        min_examples_required=min_examples_required,
    )

    reporter.info("Generating entity summarization prompt...")
    entity_summarization_prompt = create_entity_summarization_prompt(
        persona=persona, language=language,
    )

    reporter.info("Generating community reporter role...")
    community_reporter_role = await generate_community_reporter_role(
        llm, domain=domain, persona=persona, docs=doc_list
    )

    reporter.info("Generating community summarization prompt...")
    community_summarization_prompt = create_community_summarization_prompt(
        persona=persona,
        role=community_reporter_role,
        report_rating_description=community_report_ranking,
        language=language,
    )

    return (
        entity_extraction_prompt,
        entity_summarization_prompt,
        community_summarization_prompt,
    )
