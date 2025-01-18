from typing import Any
from dataclasses import dataclass

from assistant.memory.graphrag_v1.model.named import Named


@dataclass
class Entity(Named):
    type: str | None = None
    description: str | None = None
    description_embedding: list[float] | None = None
    name_embedding: list[float] | None = None
    graph_embedding: list[float] | None = None
    community_ids: list[str] | None = None
    text_unit_ids: list[str] | None = None
    document_ids: list[str] | None = None
    rank: int | None = 1
    attributes: dict[str, Any] | None = None

    @classmethod
    def from_dict(
            cls,
            d: dict[str, Any],
            id_key: str = "id",
            short_id_key: str = "short_id",
            title_key: str = 'title',
            type_key: str = "type",
            description_key: str = "description",
            description_embedding_key: str = "description_embedding",
            name_embedding_key: str = "name_embedding",
            graph_embedding_key: str = "graph_embedding",
            community_key: str = "community",
            text_unit_ids_key: str = "text_unit_ids",
            document_ids_key: str = "document_ids",
            rank_key: str = "degree",
            attributes_key: str = "attributes",
    ) -> "Entity":
        return Entity(
            id=d[id_key],
            title=d[title_key],
            short_id=d.get(short_id_key),
            type=d.get(type_key),
            description=d.get(description_key),
            name_embedding=d.get(name_embedding_key),
            description_embedding=d.get(description_embedding_key),
            graph_embedding=d.get(graph_embedding_key),
            community_ids=d.get(community_key),
            rank=d.get(rank_key, 1),
            text_unit_ids=d.get(text_unit_ids_key),
            document_ids=d.get(document_ids_key),
            attributes=d.get(attributes_key),
        )
