from typing import Any
from dataclasses import dataclass

from assistant.memory.graphrag_v1.model.named import Named


@dataclass
class CommunityReport(Named):
    community_id: str
    summary: str = ""
    full_content: str = ''
    rank: float | None = 1.0
    summary_embedding: list[float] | None = None
    full_content_embedding: list[float] | None = None
    attributes: dict[str, Any] | None = None

    @classmethod
    def from_dict(
            cls,
            d: dict[str, Any],
            id_key: str = "id",
            title_key: str = "title",
            community_id_key: str = "community_id",
            short_id_key: str = "short_id",
            summary_key: str = "summary",
            full_content_key: str = "full_content",
            rank_key: str = "rank",
            summary_embedding_key: str = "summary_embedding",
            full_content_embedding_key: str = "full_content_embedding",
            attributes_key: str = "attributes",
    ) -> "CommunityReport":
        return CommunityReport(
            id=d[id_key],
            title=d[title_key],
            community_id=d[community_id_key],
            short_id=d.get(short_id_key),
            summary=d[summary_key],
            full_content=d[full_content_key],
            rank=d[rank_key],
            summary_embedding=d.get(summary_embedding_key),
            full_content_embedding=d.get(full_content_embedding_key),
            attributes=d.get(attributes_key),
        )
