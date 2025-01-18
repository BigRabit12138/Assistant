from pathlib import Path

from assistant.memory.graphrag_v1.prompt_tune.template import COMMUNITY_REPORT_SUMMARIZATION_PROMPT

COMMUNITY_SUMMARIZATION_FILENAME = "community_report.txt"


def create_community_summarization_prompt(
        persona: str,
        role: str,
        report_rating_description: str,
        language: str,
        output_path: Path | None = None,
) -> str:
    prompt = COMMUNITY_REPORT_SUMMARIZATION_PROMPT.format(
        persona=persona,
        role=role,
        report_rating_description=report_rating_description,
        language=language,
    )

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

        output_path = output_path / COMMUNITY_SUMMARIZATION_FILENAME

        with output_path.open("wb") as file:
            file.write(prompt.encode(encoding="utf-8", errors="strict"))

    return prompt
