import os
import asyncio

from assistant.memory.graphrag_v1.index.input import load_input
from assistant.memory.graphrag_v1.index import run_pipeline_with_config
from assistant.memory.graphrag_v1.index.config import PipelineCSVInputConfig

sample_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../_sample_data/"
)


async def main():
    if "OPENAI_API_KEY" not in os.environ:
        msg = "Please set OPENAI_API_KEY environment variable to run this example"
        raise Exception(msg)

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

    dataset = dataset.head(10)

    pipeline_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./pipelines/workflows_only.yml",
    )

    tables = []
    async for table in run_pipeline_with_config(pipeline_path, dataset=dataset):
        tables.append(table)
    pipeline_result = tables[-1]

    if pipeline_result.result is not None:
        top_nodes = pipeline_result.result.head(10)
        print(
            "pipeline result\ncols:", pipeline_result.result.columns, "\n", top_nodes
        )
    else:
        print("No results!")


if __name__ == "__main__":
    asyncio.run(main())
