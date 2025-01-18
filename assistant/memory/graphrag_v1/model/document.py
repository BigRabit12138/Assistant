from typing import Any
from dataclasses import dataclass, field

from assistant.memory.graphrag_v1.model.named import Named


@dataclass
class Document(Named):
    type: str = "text"
    text_unit_ids: list[str] = field(default_factory=list)
    raw_content: str = ""
    summary: str | None = None
    summary_embedding: list[float] | None = None
    raw_content_embedding: list[float] | None = None
    attributes: dict[str, Any] | None = None

    @classmethod
    def from_dict(
            cls,
            d: dict[str, Any],
            id_key: str = "id",
            short_id_key: str = "short_id",
            title_key: str = "title",
            type_key: str = "type",
            raw_content_key: str = "raw_content",
            summary_key: str = "summary",
            summary_embedding_key: str = "summary_embedding",
            raw_content_embedding_key: str = "raw_content_embedding",
            text_units_key: str = "text_units",
            attributes_key: str = "attributes",
    ) -> "Document":
        return Document(
            id=d[id_key],
            short_id=d.get(short_id_key),
            title=d[title_key],
            type=d.get(type_key, "text"),
            raw_content=d[raw_content_key],
            summary=d.get(summary_key),
            summary_embedding=d.get(summary_embedding_key),
            raw_content_embedding=d.get(raw_content_embedding_key),
            text_unit_ids=d.get(text_units_key, []),
            attributes=d.get(attributes_key),
        )
