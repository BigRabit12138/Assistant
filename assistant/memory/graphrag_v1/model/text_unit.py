from typing import Any
from dataclasses import dataclass

from assistant.memory.graphrag_v1.model.identified import Identified


@dataclass
class TextUnit(Identified):
    text: str
    text_embedding: list[float] | None = None
    entity_ids: list[str] | None = None
    relationship_ids: list[str] | None = None
    covariate_ids: dict[str, list[str]] | None = None
    n_tokens: int | None = None
    document_ids: list[str] | None = None
    attributes: dict[str, Any] | None = None

    @classmethod
    def from_dict(
            cls,
            d: dict[str, Any],
            id_key: str = "id",
            short_id_key: str = "short_id",
            text_key: str = "text",
            text_embedding_key: str = "text_embedding",
            entities_key: str = "entity_ids",
            relationships_key: str = "relationship_ids",
            covariates_key: str = "covariate_ids",
            n_tokens_key: str = "n_tokens",
            document_ids_key: str = "document_ids",
            attributes_key: str = "attributes",
    ) -> "TextUnit":
        return TextUnit(
            id=d[id_key],
            short_id=d.get(short_id_key),
            text=d[text_key],
            text_embedding=d.get(text_embedding_key),
            entity_ids=d.get(entities_key),
            relationship_ids=d.get(relationships_key),
            covariate_ids=d.get(covariates_key),
            n_tokens=d.get(n_tokens_key),
            document_ids=d.get(document_ids_key),
            attributes=d.get(attributes_key),
        )
