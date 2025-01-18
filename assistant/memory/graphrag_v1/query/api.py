from typing import Any
from pathlib import Path
from collections.abc import AsyncGenerator

import pandas as pd

from pydantic import validate_call

from assistant.memory.graphrag_v1.model.entity import Entity
from assistant.memory.graphrag_v1.config import GraphRagConfig
from assistant.memory.graphrag_v1.index.progress.types import PrintProgressReporter
from assistant.memory.graphrag_v1.query.structured_search.base import SearchResult
from assistant.memory.graphrag_v1.vector_stores.lancedb import LanceDBVectorStore
from assistant.memory.graphrag_v1.vector_stores.typing import (
    VectorStoreType,
    VectorStoreFactory,
)
from assistant.memory.graphrag_v1.query.factories import (
    get_local_search_engine,
    get_global_search_engine,
)
from assistant.memory.graphrag_v1.query.indexer_adapters import (
    read_indexer_reports,
    read_indexer_entities,
    read_indexer_text_units,
    read_indexer_covariates,
    read_indexer_relationships,
)
from assistant.memory.graphrag_v1.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)

reporter = PrintProgressReporter("")


@validate_call(config={"arbitrary_types_allowed": True})
async def global_search(
        config: GraphRagConfig,
        nodes: pd.DataFrame,
        entities: pd.DataFrame,
        community_reports: pd.DataFrame,
        community_level: int,
        response_type: str,
        query: str,
) -> tuple[
    str | dict[str, Any] | list[dict[str, Any]],
    str | list[pd.DataFrame] | dict[str, pd.DataFrame],
]:
    reports = read_indexer_reports(community_reports, nodes, community_level)
    _entities = read_indexer_entities(nodes, entities, community_level)
    search_engine = get_global_search_engine(
        config,
        reports=reports,
        entities=_entities,
        response_type=response_type
    )
    result: SearchResult = await search_engine.asearch(query=query)
    response = result.response
    context_data = _reformat_context_data(result.context_data)
    return response, context_data


@validate_call(config={"arbitrary_types_allowed": True})
async def global_search_streaming(
        config: GraphRagConfig,
        nodes: pd.DataFrame,
        entities: pd.DataFrame,
        community_reports: pd.DataFrame,
        community_level: int,
        response_type: str,
        query: str,
) -> AsyncGenerator:
    reports = read_indexer_reports(community_reports, nodes, community_level)
    _entities = read_indexer_entities(nodes, entities, community_level)
    search_engine = get_global_search_engine(
        config,
        reports,
        entities=_entities,
        response_type=response_type,
    )
    search_result = search_engine.astream_search(query=query)

    context_data = None
    get_context_data = True
    async for stream_chunk in search_result:
        if get_context_data:
            context_data = _reformat_context_data(stream_chunk)
            yield context_data
            get_context_data = False
        else:
            yield stream_chunk


@validate_call(config={"arbitrary_types_allowed": True})
async def local_search(
        config: GraphRagConfig,
        nodes: pd.DataFrame,
        entities: pd.DataFrame,
        community_reports: pd.DataFrame,
        text_units: pd.DataFrame,
        relationships: pd.DataFrame,
        covariates: pd.DataFrame | None,
        community_level: int,
        response_type: str,
        query: str,
) -> tuple[
    str | dict[str, Any] | list[dict[str, Any]],
    str | list[pd.DataFrame] | dict[str, pd.DataFrame],
]:
    vector_store_args = (
        config.embeddings.vector_store if config.embeddings.vector_store else {}
    )
    reporter.info(f"Vector Store Args: {vector_store_args}")
    vector_store_type = vector_store_args.get("type", VectorStoreType.LanceDB)
    _entities = read_indexer_entities(nodes, entities, community_level)
    lancedb_dir = Path(config.storage.base_dir) / "lancedb"
    vector_store_args.update({"db_uri": str(lancedb_dir)})
    description_embedding_store = _get_embedding_description_store(
        entities=_entities,
        vector_store_type=vector_store_type,
        config_args=vector_store_args,
    )
    _covariates = read_indexer_covariates(covariates) if covariates is not None else []

    search_engine = get_local_search_engine(
        config=config,
        reports=read_indexer_reports(community_reports, nodes, community_level),
        text_units=read_indexer_text_units(text_units),
        entities=_entities,
        relationships=read_indexer_relationships(relationships),
        covariates={"claims": _covariates},
        description_embedding_store=description_embedding_store,
        response_type=response_type,
    )

    result: SearchResult = await search_engine.asearch(query=query)
    response = result.response
    context_data = _reformat_context_data(result.context_data)
    return response, context_data


@validate_call(config={"arbitrary_types_allowed": True})
async def local_search_streaming(
        config: GraphRagConfig,
        nodes: pd.DataFrame,
        entities: pd.DataFrame,
        community_reports: pd.DataFrame,
        text_units: pd.DataFrame,
        relationships: pd.DataFrame,
        covariates: pd.DataFrame | None,
        community_level: int,
        response_type: str,
        query: str,
) -> AsyncGenerator:
    vector_store_args = (
        config.embeddings.vector_store if config.embeddings.vector_store else {}
    )
    reporter.info(f"Vector Store Args: {vector_store_args}")
    vector_store_type = vector_store_args.get("type", VectorStoreType.LanceDB)
    _entities = read_indexer_entities(nodes, entities, community_level)
    lancedb_dir = Path(config.storage.base_dir) / "lancedb"
    vector_store_args.update({"db_uri": str(lancedb_dir)})
    description_embedding_store = _get_embedding_description_store(
        entities=_entities,
        vector_store_type=vector_store_type,
        config_args=vector_store_args,
    )
    _covariates = read_indexer_covariates(covariates) if covariates is not None else []

    search_engine = get_local_search_engine(
        config=config,
        reports=read_indexer_reports(community_reports, nodes, community_level),
        text_units=read_indexer_text_units(text_units),
        entities=_entities,
        relationships=read_indexer_relationships(relationships),
        covariates={"claims": _covariates},
        description_embedding_store=description_embedding_store,
        response_type=response_type,
    )

    search_result = search_engine.astream_search(query=query)

    context_data = None
    get_context_data = True
    async for stream_chunk in search_result:
        if get_context_data:
            context_data = _reformat_context_data(stream_chunk)
            yield context_data
            get_context_data = False
        else:
            yield stream_chunk


def _get_embedding_description_store(
        entities: list[Entity],
        vector_store_type: str = VectorStoreType.LanceDB,
        config_args: dict | None = None,
):
    if not config_args:
        config_args = {}

    collection_name = config_args.get(
        "query_collection_name", "entity_description_embeddings"
    )
    config_args.update({"collection_name": collection_name})
    description_embedding_store = VectorStoreFactory.get_vector_store(
        vector_store_type=vector_store_type,
        kwargs=config_args,
    )
    description_embedding_store.connect(**config_args)

    if config_args.get("overwrite", True):
        store_entity_semantic_embeddings(
            entities=entities,
            vectorstore=description_embedding_store,
        )
    else:
        description_embedding_store = LanceDBVectorStore(
            collection_name=collection_name,
        )
        description_embedding_store.connect(
            db_uri=config_args.get("db_uri", "./lancedb")
        )
        description_embedding_store.document_collection = (
            description_embedding_store.db_connection.open_table(
                description_embedding_store.collection_name
            )
        )

    return description_embedding_store


def _reformat_context_data(context_data: dict) -> dict:
    final_format = {
        "reports": [],
        "entities": [],
        "relationships": [],
        "claims": [],
        "sources": [],
    }
    for key in context_data:
        records = context_data[key].to_dict(orient="records")
        if len(records) < 1:
            continue
        final_format[key] = records

    return final_format
