from pathlib import Path

from assistant.memory.graphrag_v1.index.progress import PrintProgressReporter
from assistant.memory.graphrag_v1.prompt_tune.generator import MAX_TOKEN_COUNT
from assistant.memory.graphrag_v1.prompt_tune.loader import (
    MIN_CHUNK_SIZE,
    read_config_parameters,
)
from assistant.memory.graphrag_v1.prompt_tune import api
from assistant.memory.graphrag_v1.prompt_tune.generator.community_report_summarization import (
    COMMUNITY_SUMMARIZATION_FILENAME,
)
from assistant.memory.graphrag_v1.prompt_tune.generator.entity_extraction_prompt import (
    ENTITY_EXTRACTION_FILENAME
)
from assistant.memory.graphrag_v1.prompt_tune.generator.entity_summarization_prompt import (
    ENTITY_SUMMARIZATION_FILENAME,
)
from assistant.memory.graphrag_v1.prompt_tune.types import DocSelectionType


async def prompt_tune(
        config: str,
        root: str,
        domain: str,
        selection_method: DocSelectionType = DocSelectionType.RANDOM,
        limit: int = 15,
        max_tokens: int = MAX_TOKEN_COUNT,
        chunk_size: int = MIN_CHUNK_SIZE,
        language: str | None = None,
        skip_entity_types: bool = False,
        output: str = "prompts",
        n_subset_max: int = 300,
        k: int = 15,
        min_examples_required: int = 2,
):
    reporter = PrintProgressReporter("")
    graph_config = read_config_parameters(root, reporter, config)

    prompts = await api.generate_indexing_prompts(
        config=graph_config,
        root=root,
        chunk_size=chunk_size,
        limit=limit,
        selection_method=selection_method,
        domain=domain,
        language=language,
        max_tokens=max_tokens,
        skip_entity_types=skip_entity_types,
        min_examples_required=min_examples_required,
        n_subset_max=n_subset_max,
        k=k,
    )

    output_path = Path(output)

    if output_path:
        reporter.info(f"Writing prompts to {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        entity_extraction_prompt_path = output_path / ENTITY_EXTRACTION_FILENAME
        entity_summarization_prompt_path = output_path / ENTITY_SUMMARIZATION_FILENAME
        community_summarization_prompt_path = (
            output_path / COMMUNITY_SUMMARIZATION_FILENAME
        )

        with entity_extraction_prompt_path.open("wb") as file:
            file.write(prompts[0].encode('utf-8', errors="strict"))
        with entity_summarization_prompt_path.open("wb") as file:
            file.write(prompts[1].encode(encoding="utf-8", errors="strict"))
        with community_summarization_prompt_path.open("wb") as file:
            file.write(prompts[2].encode(encoding="utf-8", errors="strict"))
