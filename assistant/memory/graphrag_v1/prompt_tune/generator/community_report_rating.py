from assistant.memory.graphrag_v1.llm.types.llm_types import CompletionLLM
from assistant.memory.graphrag_v1.prompt_tune.prompt import GENERATE_REPORT_RATING_PROMPT


async def generate_community_report_rating(
        llm: CompletionLLM,
        domain: str,
        persona: str,
        docs: str | list[str],
) -> str:
    docs_str = " ".join(docs) if isinstance(docs, list) else docs
    domain_prompt = GENERATE_REPORT_RATING_PROMPT.format(
        domain=domain, persona=persona, input_text=docs_str
    )

    response = await llm(domain_prompt)

    return str(response.output).strip()
