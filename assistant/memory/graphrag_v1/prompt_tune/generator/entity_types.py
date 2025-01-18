from assistant.memory.graphrag_v1.llm.types.llm_types import CompletionLLM
from assistant.memory.graphrag_v1.prompt_tune.generator.defaults import DEFAULT_TASK
from assistant.memory.graphrag_v1.prompt_tune.prompt.entity_types import (
    ENTITY_TYPE_GENERATION_PROMPT,
    ENTITY_TYPE_GENERATION_JSON_PROMPT,
)


async def generate_entity_types(
        llm: CompletionLLM,
        domain: str,
        persona: str,
        docs: str | list[str],
        task: str = DEFAULT_TASK,
        json_mode: bool = False,
) -> str | list[str]:
    formatted_task = task.format(domain=domain)

    docs_str = "\n".join(docs) if isinstance(docs, list) else docs

    entity_types_prompt = (
        ENTITY_TYPE_GENERATION_JSON_PROMPT
        if json_mode
        else ENTITY_TYPE_GENERATION_PROMPT
    ).format(task=formatted_task, input_text=docs_str)

    history = [{"role": "system", "content": persona}]

    response = await llm(entity_types_prompt, history=history, json=json_mode)

    if json_mode:
        return (response.json or {}).get("entity_types", [])
    return str(response.output)
