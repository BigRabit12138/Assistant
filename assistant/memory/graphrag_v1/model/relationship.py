from typing import Any
from dataclasses import dataclass

from assistant.memory.graphrag_v1.model.identified import Identified


@dataclass
class Relationship(Identified):
    source: str
    target: str
    weight: float | None = 1.0
    description: str | None = None
    description_embedding: list[float] | None = None
    text_unit_ids: list[str] | None = None
    document_ids: list[str] | None = None
    attributes: dict[str, Any] | None = None

    @classmethod
    def from_dict(
            cls,
            d: dict[str, Any],
            id_key: str = "id",
            short_id_key: str = "short_id",
            source_key: str = "source",
            target_key: str = "target",
            description_key: str = "description",
            weight_key: str = "weight",
            text_unit_ids_key: str = "text_unit_ids",
            document_ids_key: str = "document_ids",
            attributes_key: str = "attributes",
    ) -> "Relationship":
        return Relationship(
            id=d[id_key],
            short_id=d.get(short_id_key),
            source=d[source_key],
            target=d[target_key],
            description=d.get(description_key),
            weight=d.get(weight_key, 1.0),
            text_unit_ids=d.get(text_unit_ids_key),
            document_ids=d.get(document_ids_key),
            attributes=d.get(attributes_key),
        )
