import os
import asyncio

import pandas as pd

from assistant.memory.graphrag_v1.index import run_pipeline_with_config

pipeline_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
)


async def run():
    dataset = _load_dataset_some_unique_way()
    config = pipeline_file

    outputs = []

    async for output in run_pipeline_with_config(
        config_or_path=config,
        dataset=dataset,
    ):
        outputs.append(output)
    pipeline_result = outputs[-1]

    if pipeline_result.result is not None:
        print(pipeline_result.result)
    else:
        print("No results!")


def _load_dataset_some_unique_way() -> pd.DataFrame:
    return pd.DataFrame([
        {"col1": 2, "col2": 4},
        {"col1": 5, "col2": 10}
    ])


if __name__ == "__main__":
    asyncio.run(run())
