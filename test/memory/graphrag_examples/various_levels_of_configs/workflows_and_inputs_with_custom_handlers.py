import os
import asyncio

from typing import Any

from datashaper import NoopWorkflowCallbacks, Progress

from assistant.memory.graphrag_v1.index import run_pipeline_with_config
from assistant.memory.graphrag_v1.index.cache import InMemoryCache, PipelineCache
from assistant.memory.graphrag_v1.index.storage import MemoryPiplineStorage


async def main():
    if "OPENAI_API_KEY" not in os.environ:
        msg = "Please set OPENAI_API_KEY environment variable to run this example"
        raise Exception(msg)

    pipeline_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./pipelines/workflows_and_inputs.yml",
    )

    custom_storage = ExampleStorage()
    custom_reporter = ExampleReporter()
    custom_cache = ExampleCache()

    tables = []
    async for table in run_pipeline_with_config(
        pipeline_path,
        storage=custom_storage,
        callbacks=custom_reporter,
        cache=custom_cache,
    ):
        tables.append(table)
    pipeline_result = tables[-1]

    if pipeline_result.result is not None:
        top_nodes = pipeline_result.result.head(10)
        print("pipeline result", top_nodes)
    else:
        print("No results!")


class ExampleStorage(MemoryPiplineStorage):
    async def get(
            self,
            key: str,
            as_bytes: bool | None = None,
            encoding: str | None = None,
    ) -> Any:
        print(f"ExampleStorage.get {key}")
        return await super().get(key, as_bytes)

    async def set(
            self,
            key: str,
            value: str | bytes | None,
            encoding: str | None = None,
    ) -> None:
        print(f"ExampleStorage.set {key}")
        return await super().set(key, value)

    async def has(self, key: str) -> bool:
        print(f"ExampleStorage.has {key}")
        return await super().has(key)

    async def delete(self, key: str) -> None:
        print(f"ExampleStorage.delete {key}")
        return await super().delete(key)

    async def clear(self) -> None:
        print("ExampleStorage.clear")
        return await super().clear()


class ExampleCache(InMemoryCache):
    async def get(self, key: str) -> Any:
        print(f"ExampleCache.get {key}")
        return await super().get(key)

    async def set(
            self,
            key: str,
            value: Any,
            debug_data: dict | None = None
    ) -> None:
        print(f"ExampleCache.set {key}")
        return await super().set(key, value, debug_data)

    async def has(self, key: str) -> bool:
        print(f"ExampleCache.has {key}")
        return await super().has(key)

    async def delete(self, key: str) -> None:
        print(f"ExampleCache.delete {key}")
        return await super().delete(key)

    async def clear(self) -> None:
        print("ExampleCache.clear")
        return await super().clear()

    async def child(self, name: str) -> PipelineCache:
        print(f"ExampleCache.child {name}")
        return ExampleCache(name)


class ExampleReporter(NoopWorkflowCallbacks):
    def progress(self, progress: Progress):
        print(f"ExampleReporter.progress: {progress}")

    def error(self, message: str, details: dict[str, Any] | None = None):
        print(f"ExampleReporter.error: {message}")

    def warning(self, message: str, details: dict[str, Any] | None = None):
        print(f"ExampleReporter.warning: {message}")

    def log(self, message: str, details: dict[str, Any] | None = None):
        print(f"ExampleReporter.log: {message}")


if __name__ == "__main__":
    asyncio.run(main())
