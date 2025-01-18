from pathlib import Path

from assistant.memory.graphrag_v1.prompt_tune.template import ENTITY_SUMMARIZATION_PROMPT

ENTITY_SUMMARIZATION_FILENAME = "summarize_descriptions.txt"


def create_entity_summarization_prompt(
        persona: str,
        language: str,
        output_path: Path | None = None,
) -> str:
    prompt = ENTITY_SUMMARIZATION_PROMPT.format(
        persona=persona, language=language
    )

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

        output_path = output_path / ENTITY_SUMMARIZATION_FILENAME

        with output_path.open("wb") as file:
            file.write(prompt.encode(encoding="utf-8", errors="strict"))

    return prompt
