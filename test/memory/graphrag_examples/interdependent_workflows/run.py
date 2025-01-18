import os
import asyncio

import pandas as pd

from assistant.memory.graphrag_v1.index import (
    run_pipeline,
    run_pipeline_with_config,
)
from assistant.memory.graphrag_v1.index.config import PipelineWorkflowReference

dataset = pd.DataFrame([
    {"type": "A", "col1": 2, "col2": 4},
    {"type": "A", "col1": 5, "col2": 10},
    {"type": "A", "col1": 15, "col2": 26},
    {"type": "B", "col1": 6, "col2": 25},
])


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
            name="aggregate_workflow",
            steps=[
                {
                    "verb": "aggregate",
                    "args": {
                        "groupby": "type",
                        "column": "col_multiplied",
                        "to": "aggregated_output",
                        "operation": "sum",
                    },
                    "input": {
                        "source": "workflow:derive_workflow",
                    },
                }
            ],
        ),
        PipelineWorkflowReference(
            name="derive_workflow",
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
            ],
        ),
    ]

    tables = []
    async for table in run_pipeline(dataset=dataset, workflows=workflows):
        tables.append(table)

    pipeline_result = tables[-1]

    if pipeline_result.result is not None:
        print(pipeline_result.result)
    else:
        print("No results!")


if __name__ == "__main__":
    asyncio.run(run_python())
    asyncio.run(run_with_config())

