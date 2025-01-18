from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from assistant.memory.graphrag_v1.model.types import TextEmbedder

DEFAULT_VECTOR_SIZE: int = 1536


@dataclass
class VectorStoreDocument:
    id: str | int
    text: str | None
    vector: list[float] | None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreSearchResult:
    document: VectorStoreDocument
    score: float


class BaseVectorStore(ABC):
    def __init__(
            self,
            collection_name: str,
            db_connection: Any | None = None,
            document_collection: Any | None = None,
            query_filter: Any | None = None,
            **kwargs: Any,
    ):
        self.collection_name = collection_name
        self.db_connection = db_connection
        self.document_collection = document_collection
        self.query_filter = query_filter
        self.kwargs = kwargs

    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def load_documents(
            self,
            documents: list[VectorStoreDocument],
            overwrite: bool = True,
    ) -> None:
        pass

    @abstractmethod
    def similarity_search_by_vector(
            self,
            query_embedding: list[float],
            k: int = 10,
            **kwargs: Any,
    ) -> list[VectorStoreSearchResult]:
        pass

    @abstractmethod
    def similarity_search_by_text(
            self,
            text: str,
            text_embedder: TextEmbedder,
            k: int = 10,
            **kwargs: Any,
    ) -> list[VectorStoreSearchResult]:
        pass

    @abstractmethod
    def filter_by_id(
            self,
            include_ids: list[str] | list[int]
    ) -> Any:
        pass
