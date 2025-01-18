import os
import asyncio

import tiktoken
import pandas as pd

from assistant.memory.graphrag_v1.query.llm.oai.typing import OpenaiApiType
from assistant.memory.graphrag_v1.query.llm.oai.chat_openai import ChatOpenAI
from assistant.memory.graphrag_v1.query.structured_search.global_search.search import GlobalSearch
from assistant.memory.graphrag_v1.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from assistant.memory.graphrag_v1.query.indexer_adapters import (
    read_indexer_reports,
    read_indexer_entities,
)

api_key = os.environ["GRAPHRAG_API_KEY"]
llm_model = os.environ["GRAPHRAG_LLM_MODEL"]

llm = ChatOpenAI(
    api_key=api_key,
    model=llm_model,
    api_type=OpenaiApiType.OpenAI,
    max_retries=20,
)
token_encoder = tiktoken.get_encoding("cl100k_base")

INPUT_DIR = "./inputs/operation dulce"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
COMMUNITY_LEVEL = 2

entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
print(f"Total report count: {len(report_df)}")
print(
    f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
)
report_df.head()

context_builder = GlobalCommunityContext(
    community_reports=reports,
    entities=entities,
    token_encoder=token_encoder,
)

context_builder_params = {
    "use_community_summary": False,
    "shuffle_data": True,
    "include_community_rank": True,
    "min_community_rank": 0,
    "community_rank_name": "rank",
    "include_community_weight": True,
    "community_weight_name": "occurrence weight",
    "normalize_community_weight": True,
    "max_tokens": 12_000,
    "context_name": "Reports",
}

map_llm_params = {
    "max_tokens": 1000,
    "temperature": 0.0,
    "response_format": {"type": "json_object"},
}

reduce_llm_params = {
    "max_tokens": 2000,
    "temperature": 0.0,
}

search_engine = GlobalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    max_data_tokens=12_000,
    map_llm_params=map_llm_params,
    reduce_llm_params=reduce_llm_params,
    allow_general_knowledge=False,
    json_mode=True,
    context_builder_params=context_builder_params,
    concurrent_coroutines=32,
    response_type="multiple paragraphs",
)
result = asyncio.run(search_engine.asearch(
    "What is the major conflict in this story and who are protagonist and antagonist?"
))

print(result.response)

print(result.context_data["reports"])
print(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}")
