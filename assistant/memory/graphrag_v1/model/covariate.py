from typing import Any
from dataclasses import dataclass

from assistant.memory.graphrag_v1.model.identified import Identified


@dataclass
class Covariate(Identified):
    subject_id: str
    subject_type: str = "entity"
    covariate_type: str = "claim"
    text_unit_ids: list[str] | None = None
    document_ids: list[str] | None = None
    attributes: dict[str, Any] | None = None

    @classmethod
    def from_dict(
            cls,
            d: dict[str, Any],
            id_key: str = "id",
            subject_id_key: str = "subject_id",
            subject_type_key: str = "subject_type",
            covariate_type_key: str = "covariate_type",
            short_id_key: str = "short_id",
            text_unit_ids_key: str = "text_unit_ids",
            document_ids_key: str = "document_ids",
            attributes_key: str = "attributes",
    ) -> "Covariate":
        return Covariate(
            id=d[id_key],
            short_id=d.get(short_id_key),
            subject_id=d[subject_id_key],
            subject_type=d.get(subject_type_key, "entity"),
            covariate_type=d.get(covariate_type_key, "claim"),
            text_unit_ids=d.get(text_unit_ids_key),
            document_ids=d.get(document_ids_key),
            attributes=d.get(attributes_key),
        )
