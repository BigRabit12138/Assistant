import asyncio
import json

from assistant.memory.graphrag_v1.llm.types.llm_types import CompletionLLM
from assistant.memory.graphrag_v1.prompt_tune.prompt import (
    ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
    ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT,
    UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
)

MAX_EXAMPLES = 5


async def generate_entity_relationship_examples(
        llm: CompletionLLM,
        persona: str,
        entity_types: str | list[str] | None,
        docs: str | list[str],
        language: str,
        json_mode: bool = False,
) -> list[str]:
    docs_list = [docs] if isinstance(docs, str) else docs
    history = [{"role": "system", "content": persona}]

    if entity_types:
        entity_types_str = (
            entity_types if isinstance(entity_types, str) else ", ".join(entity_types)
        )

        messages = [
            (
                ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
                if json_mode
                else ENTITY_RELATIONSHIPS_GENERATION_PROMPT
            ).format(entity_types=entity_types_str, input_text=doc, language=language)
            for doc in docs_list
        ]
    else:
        messages = [
            UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT.format(
                input_text=doc, language=language
            )
            for doc in docs_list
        ]

    messages = messages[: MAX_EXAMPLES]

    tasks = [llm(message, history=history, json=json_mode) for message in messages]

    responses = await asyncio.gather(*tasks)

    return [
        json.dumps(response.json or "") if json_mode else str(response.output)
        for response in responses
    ]
