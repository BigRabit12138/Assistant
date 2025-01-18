import os
import asyncio

from assistant.memory.graphrag_v1.index import run_pipeline_with_config


async def main():
    if "OPENAI_API_KEY" not in os.environ:
        msg = "Please set OPENAI_API_KEY environment variable to run this example"
        raise Exception(msg)

    pipeline_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./pipelines/workflows_and_inputs.yml",
    )

    tables = []
    async for table in run_pipeline_with_config(pipeline_path):
        tables.append(table)
    pipeline_result = tables[-1]

    if pipeline_result.result is not None:
        top_nodes = pipeline_result.result.head(10)
        print("pipeline result", top_nodes)
    else:
        print("No results!")


if __name__ == "__main__":
    asyncio.run(main())
