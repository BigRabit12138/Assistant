from assistant.memory.graphrag_v1.llm.types.llm_types import CompletionLLM
from assistant.memory.graphrag_v1.prompt_tune.prompt import GENERATE_COMMUNITY_REPORTER_ROLE_PROMPT


async def generate_community_reporter_role(
        llm: CompletionLLM,
        domain: str,
        persona: str,
        docs: str | list[str],
) -> str:
    docs_str = " ".join(docs) if isinstance(docs, list) else docs
    domain_prompt = GENERATE_COMMUNITY_REPORTER_ROLE_PROMPT.format(
        domain=domain, persona=persona, input_text=docs_str
    )

    response = await llm(domain_prompt)

    return str(response.output)
