from enum import Enum

from assistant.memory.graphrag_v1.model import (
    Entity,
    Relationship,
)
from assistant.memory.graphrag_v1.query.input.retrieval.entities import (
    get_entity_by_key,
    get_entity_by_name,
)
from assistant.memory.graphrag_v1.vector_stores import BaseVectorStore
from assistant.memory.graphrag_v1.query.llm.base import BaseTextEmbedding


class EntityVectorStoreKey(str, Enum):
    ID = "id"
    TITLE = "title"

    @staticmethod
    def from_string(value: str) -> "EntityVectorStoreKey":
        if value == "id":
            return EntityVectorStoreKey.ID
        if value == "title":
            return EntityVectorStoreKey.TITLE

        msg = f"Invalid EntityVectorStoreKey: {value}"
        raise ValueError(msg)


def map_query_to_entities(
        query: str,
        text_embedding_vectorstore: BaseVectorStore,
        text_embedder: BaseTextEmbedding,
        all_entities: list[Entity],
        embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
        include_entity_names: list[str] | None = None,
        exclude_entity_names: list[str] | None = None,
        k: int = 10,
        over_sample_scaler: int = 2,
) -> list[Entity]:
    if include_entity_names is None:
        include_entity_names = []

    if exclude_entity_names is None:
        exclude_entity_names = []

    matched_entities = []

    if query != "":
        search_results = text_embedding_vectorstore.similarity_search_by_text(
            text=query,
            text_embedder=lambda t: text_embedder.embed(t),
            k=k * over_sample_scaler,
        )
        for result in search_results:
            matched = get_entity_by_key(
                entities=all_entities,
                key=embedding_vectorstore_key,
                value=result.document.id,
            )
            if matched:
                matched_entities.append(matched)
    else:
        all_entities.sort(key=lambda x: x.rank if x.rank else 0, reverse=True)
        matched_entities = all_entities[: k]

    if exclude_entity_names:
        matched_entities = [
            entity
            for entity in matched_entities
            if entity.title not in exclude_entity_names
        ]

    included_entities = []
    for entity_name in include_entity_names:
        included_entities.extend(
            get_entity_by_name(all_entities, entity_name)
        )
    return included_entities + matched_entities


def find_nearest_neighbors_by_graph_embeddings(
        entity_id: str,
        graph_embedding_vectorstore: BaseVectorStore,
        all_entities: list[Entity],
        exclude_entity_names: list[str] | None = None,
        embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
        k: int = 10,
        over_sample_scaler: int = 2,
) -> list[Entity]:
    if exclude_entity_names is None:
        exclude_entity_names = []

    query_entity = get_entity_by_key(
        entities=all_entities, key=embedding_vectorstore_key, value=entity_id
    )
    query_embedding = query_entity.graph_embedding if query_entity else None

    if query_embedding:
        matched_entities = []
        search_results = graph_embedding_vectorstore.similarity_search_by_vector(
            query_embedding=query_embedding, k=k * over_sample_scaler
        )
        for result in search_results:
            matched = get_entity_by_key(
                entities=all_entities,
                key=embedding_vectorstore_key,
                value=result.document.id,
            )
            if matched:
                matched_entities.append(matched)

        if exclude_entity_names:
            matched_entities = [
                entity
                for entity in matched_entities
                if entity.title not in exclude_entity_names
            ]
        matched_entities.sort(key=lambda x: x.rank, reverse=True)
        return matched_entities[: k]
    return []


def find_nearest_neighbors_by_entity_rank(
        entity_name: str,
        all_entities: list[Entity],
        all_relationships: list[Relationship],
        exclude_entity_names: list[str] | None = None,
        k: int | None = 10,
) -> list[Entity]:
    if exclude_entity_names is None:
        exclude_entity_names = []
    entity_relationships = [
        rel
        for rel in all_relationships
        if rel.source == entity_name or rel.target == entity_name
    ]
    source_entity_names = {rel.source for rel in entity_relationships}
    target_entity_names = {rel.target for rel in entity_relationships}
    related_entity_names = (
        source_entity_names.union(target_entity_names)
    ).difference(
        set(exclude_entity_names)
    )
    top_relations = [
        entity for entity in all_entities if entity.title in related_entity_names
    ]
    top_relations.sort(key=lambda x: x.rank if x.rank else 0, reverse=True)
    if k:
        return top_relations[: k]
    return top_relations


