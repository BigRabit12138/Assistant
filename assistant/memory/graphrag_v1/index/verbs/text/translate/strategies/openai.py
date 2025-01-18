import logging
import traceback

from typing import Any

from datashaper import VerbCallbacks

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.llm import CompletionLLM
from assistant.memory.graphrag_v1.index.llm import load_llm
from assistant.memory.graphrag_v1.config.enums import LLMType
from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.text_splitting import TokenTextSplitter
from assistant.memory.graphrag_v1.index.verbs.text.translate.strategies.typing import TextTranslationResult
from assistant.memory.graphrag_v1.index.verbs.text.translate.strategies.defaults import (TRANSLATION_PROMPT as
                                                                                      DEFAULT_TRANSLATION_PROMPT)
log = logging.getLogger(__name__)


async def run(
        input_: str | list[str],
        args: dict[str, Any],
        callbacks: VerbCallbacks,
        pipeline_cache: PipelineCache,
) -> TextTranslationResult:
    llm_config = args.get("llm", {"type": LLMType.StaticResponse})
    llm_type = llm_config.get("type", LLMType.StaticResponse)
    llm = load_llm(
        "text_translation",
        llm_type,
        callbacks,
        pipeline_cache,
        llm_config,
        chat_only=True,
    )
    language = args.get("language", "English")
    prompt = args.get("prompt")
    chunk_size = args.get("chunk_size", defaults.CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", defaults.CHUNK_OVERLAP)

    input_ = [input_] if isinstance(input_, str) else input_
    return TextTranslationResult(
        translations=[
            await _translate_text(
                text, language, prompt, llm, chunk_size, chunk_overlap, callbacks
            )
            for text in input_
        ]
    )


async def _translate_text(
        text: str,
        language: str,
        prompt: str | None,
        llm: CompletionLLM,
        chunk_size: int,
        chunk_overlap: int,
        callbacks: VerbCallbacks,
) -> str:
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    out = ""
    chunks = splitter.split_text(text)
    for chunk in chunks:
        try:
            result = await llm(
                chunk,
                history=[
                    {
                        "role": "system",
                        "content": (prompt or DEFAULT_TRANSLATION_PROMPT),
                    }
                ],
                variables={"language": language},
            )
            out += result.output or ""
        except Exception as e:
            log.exception("error translating text.")
            callbacks.error("Error translating text", e, traceback.format_exc())
            out += ""

    return out
