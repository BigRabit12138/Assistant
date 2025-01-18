from typing import Any

from datashaper import VerbCallbacks

from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.verbs.text.translate.strategies.typing import TextTranslationResult


async def run(
        input_: str | list[str],
        _args: dict[str, Any],
        _reporter: VerbCallbacks,
        _cache: PipelineCache,
) -> TextTranslationResult:
    input_ = [input_] if isinstance(input_, str) else input_
    return TextTranslationResult(translations=[_translate_text(text) for text in input_])


def _translate_text(text: str) -> str:
    return f"{text} translated"