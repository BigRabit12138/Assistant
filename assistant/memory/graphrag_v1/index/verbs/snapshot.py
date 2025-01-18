from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)

from assistant.memory.graphrag_v1.index.storage import PipelineStorage


@verb(name="snapshot")
async def snapshot(
        input: VerbInput,
        name: str,
        formats: list[str],
        storage: PipelineStorage,
        **_kwargs: dict,
) -> TableContainer:
    data = input.get_input()

    for fmt in formats:
        if fmt == "parquet":
            await storage.set(name + ".parquet", data.to_parquet())
        elif fmt == "json":
            await storage.set(
                name + ".json", data.to_json(orient="records", lines=True)
            )

    return TableContainer(table=data)
