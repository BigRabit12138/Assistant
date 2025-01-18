from typing import cast

import numpy as np
import pandas as pd
from datashaper import (
    VerbInput,
    TableContainer,
    NoopVerbCallbacks,
)

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.index.verbs import chunk
from assistant.memory.graphrag_v1.index.input import load_input
from assistant.memory.graphrag_v1.index.llm import load_llm_embeddings
from assistant.memory.graphrag_v1.llm.types.llm_types import EmbeddingLLM
from assistant.memory.graphrag_v1.index.progress.types import ProgressReporter
from assistant.memory.graphrag_v1.config.models.graph_rag_config import GraphRagConfig

K = 15
N_SUBSET_MAX = 300
MIN_CHUNK_SIZE = 200
MIN_CHUNK_OVERLAP = 0


async def _embed_chunks(
        text_chunks: pd.DataFrame,
        embedding_llm: EmbeddingLLM,
        n_subset_max: int = N_SUBSET_MAX,
) -> tuple[pd.DataFrame, np.ndarray]:
    sampled_text_chunks = text_chunks.sample(n=min(n_subset_max, len(text_chunks)))
    embeddings = await embedding_llm(sampled_text_chunks["chunks"].tolist())
    return text_chunks, np.array(embeddings.output)


def _sample_chunks_from_embeddings(
        text_chunks: pd.DataFrame,
        embeddings,
        k: int = K,
) -> pd.DataFrame:
    center = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - center, axis=1)
    nearest_indices = np.argsort(distances)[: k]

    return text_chunks.iloc[nearest_indices]


async def load_docs_in_chunks(
        root: str,
        config: GraphRagConfig,
        select_method: str,
        limit: int,
        reporter: ProgressReporter,
        chunk_size: int = MIN_CHUNK_SIZE,
        n_subset_max: int = N_SUBSET_MAX,
        k: int = K,
) -> list[str]:
    dataset = await load_input(config.input, reporter, root)

    input_ = VerbInput(input=TableContainer(table=dataset))
    chunk_strategy = config.chunks.resolved_strategy(defaults.ENCODING_MODEL)

    chunk_strategy["chunk_size"] = chunk_size
    chunk_strategy["chunk_overlap"] = MIN_CHUNK_OVERLAP

    dataset_chunks_table_container = chunk(
        input_,
        column="text",
        to="chunks",
        callbacks=NoopVerbCallbacks(),
        strategy=chunk_strategy
    )

    dataset_chunks = cast(pd.DataFrame, dataset_chunks_table_container.table)

    chunks_df = pd.DataFrame(dataset_chunks["chunks"].explode())

    if limit <= 0 or limit > len(chunks_df):
        limit = len(chunks_df)

    if select_method == "top":
        chunks_df = chunks_df[: limit]
    elif select_method == "random":
        chunks_df = chunks_df.sample(n=limit)
    elif select_method == "auto":
        if k is None or k <= 0:
            msg = "k must be an integer > 0"
            raise ValueError(msg)
        embeddings_llm = load_llm_embeddings(
            name="prompt_tuning_embeddings",
            llm_type=config.embeddings.resolved_strategy()["llm"]["type"],
            callbacks=NoopVerbCallbacks(),
            cache=None,
            llm_config=config.embeddings.resolved_strategy()["llm"],
        )

        chunks_df, embeddings = await _embed_chunks(
            chunks_df, embeddings_llm, n_subset_max=n_subset_max
        )

        chunks_df = _sample_chunks_from_embeddings(
            chunks_df, embeddings, k=k
        )

    return chunks_df["chunks"].tolist()
