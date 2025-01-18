from pathlib import Path

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.index.utils.tokens import num_tokens_from_string
from assistant.memory.graphrag_v1.prompt_tune.template import (
    GRAPH_EXTRACTION_PROMPT,
    EXAMPLE_EXTRACTION_TEMPLATE,
    GRAPH_EXTRACTION_JSON_PROMPT,
    UNTYPED_GRAPH_EXTRACTION_PROMPT,
    UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE,
)

ENTITY_EXTRACTION_FILENAME = "entity_extraction.txt"


def create_entity_extraction_prompt(
        entity_types: str | list[str] | None,
        docs: list[str],
        examples: list[str],
        language: str,
        max_token_count: int,
        encoding_model: str = defaults.ENCODING_MODEL,
        json_mode: bool = False,
        output_path: Path | None = None,
        min_examples_required: int = 2,
) -> str:
    prompt = (
        (GRAPH_EXTRACTION_JSON_PROMPT if json_mode else GRAPH_EXTRACTION_PROMPT)
        if entity_types else UNTYPED_GRAPH_EXTRACTION_PROMPT
    )

    if isinstance(entity_types, list):
        entity_types = ", ".join(entity_types)

    tokens_left = (
        max_token_count
        - num_tokens_from_string(prompt, encoding_name=encoding_model)
        - num_tokens_from_string(entity_types, encoding_name=encoding_model)
        if entity_types
        else 0
    )

    examples_prompt = ""

    for i, output in enumerate(examples):
        input_ = docs[i]
        example_formatted = (
            EXAMPLE_EXTRACTION_TEMPLATE.format(
                n=i + 1, input_text=input_, entity_types=entity_types, output=output
            )
            if entity_types
            else UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE.format(
                n=i + 1, input_text=input_, output=output
            )
        )

        example_tokens = num_tokens_from_string(
            example_formatted, encoding_name=encoding_model
        )

        if i >= min_examples_required and example_tokens > tokens_left:
            break

        examples_prompt += example_formatted
        tokens_left -= example_tokens

    prompt = (
        prompt.format(
            entity_types=entity_types, examples=examples_prompt, language=language
        )
        if entity_types
        else prompt.format(examples=examples_prompt, language=language)
    )

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

        output_path = output_path / ENTITY_EXTRACTION_FILENAME

        with output_path.open("wb") as file:
            file.write(prompt.encode(encoding="utf-8", errors="strict"))

    return prompt
