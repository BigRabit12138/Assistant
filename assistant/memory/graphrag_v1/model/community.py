from typing import Any
from dataclasses import dataclass

from assistant.memory.graphrag_v1.model.named import Named


@dataclass
class Community(Named):
    level: str = ""
    entity_ids: list[str] | None = None
    relationship_ids: list[str] | None = None
    covariate_ids: dict[str, list[str]] | None = None
    attributes: dict[str, Any] | None = None

    @classmethod
    def from_dict(
            cls,
            d: dict[str, Any],
            id_key: str = "id",
            title_key: str = "title",
            short_id_key: str = "short_id",
            level_key: str = "level",
            entities_key: str = "entity_ids",
            relationships_key: str = "relationship_ids",
            covariates_key: str = "covariate_ids",
            attributes_key: str = "attributes",
    ) -> "Community":
        return Community(
            id=d[id_key],
            title=d[title_key],
            short_id=d.get(short_id_key),
            level=d[level_key],
            entity_ids=d.get(entities_key),
            relationship_ids=d.get(relationships_key),
            covariate_ids=d.get(covariates_key),
            attributes=d.get(attributes_key),
        )
