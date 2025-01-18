from assistant.memory.graphrag_v1.llm.types.llm_types import CompletionLLM
from assistant.memory.graphrag_v1.prompt_tune.prompt.domain import GENERATE_DOMAIN_PROMPT


async def generate_domain(
        llm: CompletionLLM,
        docs: str | list[str]
) -> str:
    docs_str = " ".join(docs) if isinstance(docs, list) else docs
    domain_prompt = GENERATE_DOMAIN_PROMPT.format(input_text=docs_str)

    response = await llm(domain_prompt)

    return str(response.output)
