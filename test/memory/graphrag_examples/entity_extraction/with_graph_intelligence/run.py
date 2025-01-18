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
        print(pipeline_result.result["entities"].to_list())
    else:
        print("No results!")


async def run_python():
    if "OPENAI_API_KEY" not in os.environ:
        msg = "Please set OPENAI_API_KEY environment variable to run this example"
        raise Exception(msg)

    dataset = shared_dataset.head(10)

    workflows: list[PipelineWorkflowReference] = [
        PipelineWorkflowReference(
            name="entity_extraction",
            config={
                "entity_extract": {
                    "strategy": {
                        "type": "graph_intelligence",
                        "llm": {
                            "type": "openai_chat",
                            "api_key": os.environ.get(
                                "OPENAI_API_KEY", None
                            ),
                            "model": os.environ.get(
                                "OPENAI_MODEL", "gpt-3.5-turbo"
                            ),
                            "max_tokens": os.environ.get(
                                "OPENAI_MAX_TOKENS", 2500
                            ),
                            "temperature": os.environ.get(
                                "OPENAI_TEMPERATURE", 0
                            ),
                        },
                    }
                }
            },
        )
    ]

    tables = []
    async for table in run_pipeline(
        dataset=dataset, workflows=workflows
    ):
        tables.append(table)

    pipeline_result = tables[-1]

    if pipeline_result.result is not None:
        print(pipeline_result.result["entities"].to_list())
    else:
        print("No results!")


if __name__ == "__main__":
    asyncio.run(run_python())
    asyncio.run(run_with_config())
