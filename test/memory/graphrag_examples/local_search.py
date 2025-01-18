import os
import asyncio

import tiktoken
import pandas as pd

from dotenv import load_dotenv

from assistant.memory.graphrag_v1.query.llm.oai.typing import OpenaiApiType
from assistant.memory.graphrag_v1.query.llm.oai.chat_openai import ChatOpenAI
from assistant.memory.graphrag_v1.query.llm.oai.embedding import OpenAIEmbedding
from assistant.memory.graphrag_v1.vector_stores.lancedb import LanceDBVectorStore
from assistant.memory.graphrag_v1.query.question_gen.local_gen import LocalQuestionGen
from assistant.memory.graphrag_v1.query.structured_search.local_search.search import LocalSearch
from assistant.memory.graphrag_v1.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
)
from assistant.memory.graphrag_v1.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from assistant.memory.graphrag_v1.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from assistant.memory.graphrag_v1.query.indexer_adapters import (
    read_indexer_reports,
    read_indexer_entities,
    read_indexer_covariates,
    read_indexer_text_units,
    read_indexer_relationships,
)

load_dotenv('./.env')

INPUT_DIR = "./inputs/operation dulce"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2

entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

description_embedding_store = LanceDBVectorStore(
    collection_name="entity_description_embeddings",
)
description_embedding_store.connect(db_uri=LANCEDB_URI)
entity_description_embeddings = store_entity_semantic_embeddings(
    entities=entities, vectorstore=description_embedding_store
)

print(f"Entity count: {len(entity_df)}")
entity_df.head()


relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
relationships = read_indexer_relationships(relationship_df)

print(f"Relationship count: {len(relationship_df)}")
relationship_df.head()

covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")
claims = read_indexer_covariates(covariate_df)
print(f"Claim records: {len(claims)}")
covariates = {"claims": claims}


report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
print(f"Report records: {len(report_df)}")
report_df.head()


text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
text_units = read_indexer_text_units(text_unit_df)
print(f"Text unit records: {len(text_unit_df)}")
text_unit_df.head()

api_key = os.environ["GRAPHRAG_API_KEY"]
llm_model = os.environ["GRAPHRAG_LLM_MODEL"]
embedding_model = os.environ["GRAPHRAG_EMBEDDING_MODEL"]

llm = ChatOpenAI(
    api_key=api_key,
    model=llm_model,
    api_type=OpenaiApiType.OpenAI,
    max_retries=20,
    api_base="https://api.chatanywhere.tech/v1"
)

token_encoder = tiktoken.get_encoding("cl100k_base")
text_embedder = OpenAIEmbedding(
    api_key=api_key,
    api_base="https://api.chatanywhere.tech/v1",
    api_type=OpenaiApiType.OpenAI,
    model=embedding_model,
    deployment_name=embedding_model,
    max_retries=20,
)


context_builder = LocalSearchMixedContext(
    community_reports=reports,
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    covariates=covariates,
    entity_text_embeddings=description_embedding_store,
    embedding_vectorstore_key=EntityVectorStoreKey.ID,
    text_embedder=text_embedder,
    token_encoder=token_encoder,
)


local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.1,
    "conversation_history_max_turns": 5,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 10,
    "top_k_relationships": 10,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": False,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,
    "max_tokens": 12_000,
}

llm_params = {
    "max_tokens": 2_000,
    "temperature": 0.0,
}

search_engine = LocalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    llm_params=llm_params,
    context_builder_params=local_context_params,
    response_type="multiple paragraphs",
)

result = asyncio.run(search_engine.asearch("Tell me about Agent Mercer"))
print(result.response)

question = "Tell me about Dr. Jordan Hayes"
result = asyncio.run(search_engine.asearch(question))
print(result.response)

result_en = result.context_data["entities"].head()
print(result_en)
result_re = result.context_data["relationships"].head()
print(result_re)
result_repo = result.context_data["reports"].head()
print(result_repo)
result_sour = result.context_data["sources"].head()
print(result_sour)
if "claims" in result.context_data:
    print(result.context_data["claims"].head())

question_generator = LocalQuestionGen(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    llm_params=llm_params,
    context_builder_params=local_context_params,
)

question_history = [
    "Tell me about Agent Mercer",
    "Tell me about Dr. Jordan Hayes"
]
candidate_questions = asyncio.run(question_generator.agenerate(
    question_history=question_history, context_data=None, question_count=5
))
print(candidate_questions.response)
