import os
import asyncio

from assistant.memory.graphrag_v1.index.input import load_input
from assistant.memory.graphrag_v1.index import run_pipeline_with_config
from assistant.memory.graphrag_v1.index.config import PipelineCSVInputConfig

sample_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "./../sample_data/"
)


async def run_with_config():
    dataset = await load_input(
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

    dataset = dataset.head(2)

    pipeline_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    async for result in run_pipeline_with_config(pipeline_path, dataset=dataset):
        print(f"Workflow {result.workflow} result\n: ")
        print(result.result)


if __name__ == "__main__":
    asyncio.run(run_with_config())
