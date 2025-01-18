from assistant.memory.graphrag_v1.llm.types.llm_types import CompletionLLM
from assistant.memory.graphrag_v1.prompt_tune.generator.defaults import DEFAULT_TASK
from assistant.memory.graphrag_v1.prompt_tune.prompt import GENERATE_PERSONA_PROMPT


async def generate_persona(
        llm: CompletionLLM,
        domain: str,
        task: str = DEFAULT_TASK,
) -> str:
    formatted_task = task.format(domain=domain)
    persona_prompt = GENERATE_PERSONA_PROMPT.format(sample_task=formatted_task)

    response = await llm(persona_prompt)

    return str(response.output)
