import os
import asyncio

import pandas as pd

from assistant.memory.graphrag_v1.index.config import PipelineWorkflowReference
from assistant.memory.graphrag_v1.index import (
    run_pipeline,
    run_pipeline_with_config,
)

dataset = pd.DataFrame([{"col1": 2, "col2": 4}, {"col1": 5, "col2": 10}])


async def run_with_config():
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
        print(pipeline_result.result)
    else:
        print("No results!")


async def run_python():
    workflows: list[PipelineWorkflowReference] = [
        PipelineWorkflowReference(
            steps=[
                {
                    "verb": "derive",
                    "args": {
                        "column1": "col1",
                        "column2": "col2",
                        "to": "col_multiplied",
                        "operator": "*",
                    },
                }
            ]
        ),
    ]

    tables = []
    async for table in run_pipeline(
            workflows=workflows, dataset=dataset
    ):
        tables.append(table)
    pipeline_result = tables[-1]

    if pipeline_result.result is not None:
        print(pipeline_result.result)
    else:
        print("No results!")


if __name__ == "__main__":
    asyncio.run(run_with_config())
    asyncio.run(run_python())
