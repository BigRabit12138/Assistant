from assistant.memory.graphrag_v1.llm.types.llm_types import CompletionLLM
from assistant.memory.graphrag_v1.prompt_tune.prompt import DETECT_LANGUAGE_PROMPT


async def detect_language(
        llm: CompletionLLM,
        docs: str | list[str],
) -> str:
    docs_str = " ".join(docs) if isinstance(docs, list) else docs
    language_prompt = DETECT_LANGUAGE_PROMPT.format(input_text=docs_str)

    response = await llm(language_prompt)

    return str(response.output)
