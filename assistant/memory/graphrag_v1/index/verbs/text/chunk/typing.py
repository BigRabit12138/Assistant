from dataclasses import dataclass


@dataclass
class TextChunk:
    """
    文本块
    """
    text_chunk: str
    source_doc_indices: list[int]
    n_tokens: int | None = None


ChunkInput = str | list[str] | list[tuple[str, str]]
