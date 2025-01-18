import os
import asyncio

from assistant.memory.graphrag_v1.index import (
    run_pipeline,
    run_pipeline_with_config,
)
from assistant.memory.graphrag_v1.index.config import (
    PipelineCSVInputConfig,
    PipelineWorkflowReference,
)
from assistant.memory.graphrag_v1.index.input import load_input

sample_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../sample_data/"
)

shared_dataset = asyncio.run(
    load_input(
        PipelineCSVInputConfig(
            file_pattern=".*\\.csv$",
            base_dir=sample_data_dir,
            source_column="author",
            text_column="message",
            timestamp_column="date(yyyyMMddHHmmss)",
            timestamp_format="%Y%m%d%H%M%S",
            title_column="message",
        ),
    )
)


async def run_with_config():
    dataset = shared_dataset.head(10)
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    tables = []
    async for table in run_pipeline_with_config(
        config_or_path=config_path, dataset=dataset
    ):
        tables.append(table)

    pipeline_result = tables[-1]

    if pipeline_result.result is not None:
        first_result = pipeline_result.result.head(1)
        print(f"level: {first_result['level'][0]}")
        print(f"embeddings: {first_result['embeddings'][0]}")
        print(f"entity_graph_positions: {first_result['node_positions'][0]}")
    else:
        print("No results!")


async def run_python():
    dataset = shared_dataset.head(10)

    workflows: list[PipelineWorkflowReference] = [
        PipelineWorkflowReference(
            name="entity_extraction",
            config={
                "entity_extract": {
                    "strategy": {
                        "type": "nltk",
                    }
                }
            },
        ),
        PipelineWorkflowReference(
            name="entity_graph",
            config={
                "cluster_graph": {"strategy": {"type": "leiden"}},
                "embed_graph": {
                    "strategy": {
                        "type": "node2vec",
                        "num_walks": 10,
                        "walk_length": 40,
                        "window_size": 2,
                        "iterations": 3,
                        "random_seed": 597832,
                    }
                },
                "layout_graph": {
                    "strategy": {
                        "type": "umap",
                    },
                },
            },
        ),
    ]

    tables = []
    async for table in run_pipeline(
            workflows=workflows, dataset=dataset
    ):
        tables.append(table)

    pipeline_result = tables[-1]

    if pipeline_result.result is not None:
        first_result = pipeline_result.result.head(1)
        print(f"level: {first_result['level'][0]}")
        print(f"embeddings: {first_result['embeddings'][0]}")
        print(f"entity_graph_positions: {first_result['node_positions'][0]}")
    else:
        print("No results!")


if __name__ == "__main__":
    asyncio.run(run_python())
    asyncio.run(run_with_config())
