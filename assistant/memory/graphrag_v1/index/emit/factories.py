from assistant.memory.graphrag_v1.index.typing import ErrorHandlerFn
from assistant.memory.graphrag_v1.index.storage import PipelineStorage

from assistant.memory.graphrag_v1.index.emit.types import TableEmitterType
from assistant.memory.graphrag_v1.index.emit.table_emitter import TableEmitter
from assistant.memory.graphrag_v1.index.emit.csv_table_emitter import CSVTableEmitter
from assistant.memory.graphrag_v1.index.emit.json_table_emitter import JsonTableEmitter
from assistant.memory.graphrag_v1.index.emit.parquet_table_emitter import ParquetTableEmitter


def create_table_emitter(
        emitter_type: TableEmitterType,
        storage: PipelineStorage,
        on_error: ErrorHandlerFn
) -> TableEmitter:
    match emitter_type:
        case TableEmitterType.Json:
            return JsonTableEmitter(storage)
        case TableEmitterType.Parquet:
            return ParquetTableEmitter(storage, on_error)
        case TableEmitterType.CSV:
            return CSVTableEmitter(storage)
        case _:
            msg = f"Unsupported table emitter type: {emitter_type}"
            raise ValueError(msg)


def create_table_emitters(
        emitter_types: list[TableEmitterType],
        storage: PipelineStorage,
        on_error: ErrorHandlerFn,
) -> list[TableEmitter]:
    return [
        create_table_emitter(emitter_type, storage, on_error)
        for emitter_type in emitter_types
    ]
