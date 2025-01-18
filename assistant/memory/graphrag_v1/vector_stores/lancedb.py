import json

from typing import Any

import pyarrow as pa
import lancedb as lancedb

from assistant.memory.graphrag_v1.model.types import TextEmbedder
from assistant.memory.graphrag_v1.vector_stores.base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)


class LanceDBVectorStore(BaseVectorStore):
    def connect(self, **kwargs: Any) -> Any:
        db_uri = kwargs.get("db_uri", "./lancedb")
        self.db_connection = lancedb.connect(db_uri)

    def load_documents(
            self,
            documents: list[VectorStoreDocument],
            overwrite: bool = True,
    ) -> None:
        data = [
            {
                "id": document.id,
                "text": document.text,
                "vector": document.vector,
                "attributes": json.dumps(document.attributes),
            }
            for document in documents
            if document.vector is not None
        ]

        if len(data) == 0:
            data = None

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float64())),
            pa.field("attributes", pa.string()),
        ])
        if overwrite:
            if data:
                self.document_collection = self.db_connection.create_table(
                    self.collection_name, data=data, mode="overwrite"
                )
            else:
                self.document_collection = self.db_connection.create_table(
                    self.collection_name, schema=schema, mode="overwrite"
                )
        else:
            self.document_collection = self.db_connection.open_table(
                self.collection_name
            )
            if data:
                self.document_collection.add(data)

    def filter_by_id(
            self,
            include_ids: list[str] | list[int]
    ) -> Any:
        if len(include_ids) == 0:
            self.query_filter = None
        else:
            if isinstance(include_ids[0], str):
                id_filter = ", ".join([f"'{id_}'" for id_ in include_ids])
                self.query_filter = f"id in ({id_filter})"
            else:
                self.query_filter = (
                    f"id in ({', '.join([str(id_) for id_ in include_ids])})"
                )
        return self.query_filter

    def similarity_search_by_vector(
            self,
            query_embedding: list[float],
            k: int = 10,
            **kwargs: Any,
    ) -> list[VectorStoreSearchResult]:
        if self.query_filter:
            docs = (
                self.document_collection.search(query=query_embedding)
                .where(self.query_filter, prefilter=True)
                .limit(k)
                .to_list()
            )
        else:
            docs = (
                self.document_collection.search(query=query_embedding)
                .limit(k)
                .to_list()
            )
        return [
            VectorStoreSearchResult(
                document=VectorStoreDocument(
                    id=doc["id"],
                    text=doc["text"],
                    vector=doc["vector"],
                    attributes=json.loads(doc["attributes"]),
                ),
                score=1-abs(float(doc["_distance"])),
            )
            for doc in docs
        ]

    def similarity_search_by_text(
            self,
            text: str,
            text_embedder: TextEmbedder,
            k: int = 10,
            **kwargs: Any,
    ) -> list[VectorStoreSearchResult]:
        query_embedding = text_embedder(text)
        if query_embedding:
            return self.similarity_search_by_vector(query_embedding, k)
        return []
