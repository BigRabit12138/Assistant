from enum import Enum
from typing import ClassVar

from assistant.memory.graphrag_v1.vector_stores.lancedb import LanceDBVectorStore
from assistant.memory.graphrag_v1.vector_stores.azure_ai_search import AzureAISearch


class VectorStoreType(str, Enum):
    LanceDB = "lancedb"
    AzureAISearch = "azure_ai_search"


class VectorStoreFactory:
    vector_store_types: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, vector_store_type: str, vector_store: type):
        cls.vector_store_types[vector_store_type] = vector_store

    @classmethod
    def get_vector_store(
            cls,
            vector_store_type: VectorStoreType | str,
            kwargs: dict
    ) -> LanceDBVectorStore | AzureAISearch:
        match vector_store_type:
            case VectorStoreType.LanceDB:
                return LanceDBVectorStore(**kwargs)
            case VectorStoreType.AzureAISearch:
                return AzureAISearch(**kwargs)
            case _:
                if vector_store_type in cls.vector_store_types:
                    return cls.vector_store_types[vector_store_type](**kwargs)
                msg = f"Unknown vector store type: {vector_store_type}."
                raise ValueError(msg)
