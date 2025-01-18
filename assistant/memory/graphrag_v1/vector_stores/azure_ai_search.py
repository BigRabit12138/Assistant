import json

from typing import Any

from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchIndex,
    SimpleField,
    VectorSearch,
    HnswParameters,
    SearchableField,
    SearchFieldDataType,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmMetric,
)

from assistant.memory.graphrag_v1.model.types import TextEmbedder
from assistant.memory.graphrag_v1.vector_stores.base import (
    BaseVectorStore,
    DEFAULT_VECTOR_SIZE,
    VectorStoreDocument,
    VectorStoreSearchResult,
)


class AzureAISearch(BaseVectorStore):
    index_client: SearchIndexClient

    def connect(self, **kwargs: Any) -> Any:
        url = kwargs.get("url", None)
        api_key = kwargs.get("api_key", None)
        audience = kwargs.get("audience", None)
        self.vector_size = kwargs.get("vector_size", DEFAULT_VECTOR_SIZE)

        self.vector_search_profile_name = kwargs.get(
            "vector_search_profile_name",
            "vectorSearchProfile"
        )

        if url:
            audience_arg = {"audience": audience} if audience else {}
            self.db_connection = SearchClient(
                endpoint=url,
                index_name=self.collection_name,
                credential=AzureKeyCredential(api_key)
                if api_key
                else DefaultAzureCredential(),
                **audience_arg,
            )
            self.index_client = SearchIndexClient(
                endpoint=url,
                credential=AzureKeyCredential(api_key)
                if api_key
                else DefaultAzureCredential(),
                **audience_arg,
            )
        else:
            not_supported_error = "AAISearchDBClient is not supported on local host."
            raise ValueError(not_supported_error)

    def load_documents(
            self,
            documents: list[VectorStoreDocument],
            overwrite: bool = True,
    ) -> None:

        if overwrite:
            if self.collection_name in self.index_client.list_index_names():
                self.index_client.delete_index(self.collection_name)

            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="HnswAlg",
                        parameters=HnswParameters(
                            metric=VectorSearchAlgorithmMetric.COSINE
                        ),
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name=self.vector_search_profile_name,
                        algorithm_configuration_name="HnswAlg",
                    )
                ],
            )

            index = SearchIndex(
                name=self.collection_name,
                fields=[
                    SimpleField(
                        name="id",
                        type=SearchFieldDataType.String,
                        key=True,
                    ),
                    SearchField(
                        name="vector",
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True,
                        vector_search_dimensions=self.vector_size,
                        vector_search_profile_name=self.vector_search_profile_name,
                    ),
                    SearchableField(name="text", type=SearchFieldDataType.String),
                    SimpleField(
                        name="attributes",
                        type=SearchFieldDataType.String,
                    ),
                ],
                vector_search=vector_search,
            )

            self.index_client.create_or_update_index(
                index,
            )

        batch = [
            {
                "id": doc.id,
                "vector": doc.vector,
                "text": doc.text,
                "attributes": json.dumps(doc.attributes),
            }
            for doc in documents
            if doc.vector is not None
        ]

        if batch and len(batch) > 0:
            self.db_connection.upload_documents(batch)

    def filter_by_id(
            self,
            include_ids: list[str] | list[int]
    ) -> Any:
        if include_ids is None or len(include_ids) == 0:
            self.query_filter = None
            return self.query_filter

        id_filter = ",".join([f"{id_!s}" for id_ in include_ids])
        self.query_filter = f"search.in(id, '{id_filter}', ',')"

        return self.query_filter

    def similarity_search_by_vector(
            self,
            query_embedding: list[float],
            k: int = 10,
            **kwargs: Any,
    ) -> list[VectorStoreSearchResult]:
        vectorized_query = VectorizedQuery(
            vector=query_embedding, k_nearest_neighbors=k, fields="vector"
        )

        response = self.db_connection.search(
            vector_queries=[vectorized_query],
        )

        return [
            VectorStoreSearchResult(
                document=VectorStoreDocument(
                    id=doc.get("id", ""),
                    text=doc.get("text", ""),
                    vector=doc.get("vector", []),
                    attributes=(json.loads(doc.get("attributes", "{}"))),
                ),
                score=doc["@search.score"],
            )
            for doc in response
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
            return self.similarity_search_by_vector(
                query_embedding=query_embedding, k=k
            )
        return []
