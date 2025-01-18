import os
import asyncio

import pandas as pd

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)

from assistant.memory.graphrag_v1.index.config import PipelineWorkflowReference
from assistant.memory.graphrag_v1.index import (
    run_pipeline,
    run_pipeline_with_config,
)


@verb(name="str_append")
def str_append(
        input: VerbInput,
        source_column: str,
        target_column: str,
        string_to_append: str,
        **_kwargs: dict,
):
    input_data = input.get_input()
    output_df = input_data.copy()
    output_df[target_column] = output_df[source_column].apply(
        lambda x: f"{x}{string_to_append}"
    )
    return TableContainer(table=output_df)


custom_verbs = {
    "str_append": str_append,
}

dataset = pd.DataFrame([{"col1": 2, "col2": 4}, {"col1": 5, "col2": 10}])


async def run_with_config():
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    outputs = []
    async for output in run_pipeline_with_config(
        config_or_path=config_path, dataset=dataset
    ):
        outputs.append(output)
    pipeline_result = outputs[-1]

    if pipeline_result.result is not None:
        print(pipeline_result.result)
    else:
        print("No results!")


async def run_python():
    workflows: list[PipelineWorkflowReference] = [
        PipelineWorkflowReference(
            name="my_workflow",
            steps=[
                {
                    "verb": "str_append",
                    "args": {
                        "source_column": "col1",
                        "target_column": "col_1_custom",
                        "string_to_append": " - custom verb",
                    },
                }
            ],
        ),
    ]

    outputs = []
    async for output in run_pipeline(
        dataset=dataset,
        workflows=workflows,
        additional_verbs=custom_verbs,
    ):
        outputs.append(output)

    pipeline_result = next(
        (output for output in outputs if output.workflow == "my_workflow"), None
    )

    if pipeline_result is not None and pipeline_result.result is not None:
        print(pipeline_result.result)
    else:
        print("No results!")


if __name__ == "__main__":
    asyncio.run(run_with_config())
    asyncio.run(run_python())

