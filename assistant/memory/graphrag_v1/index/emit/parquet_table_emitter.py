import logging
import traceback

import pandas as pd

from pyarrow.lib import ArrowInvalid, ArrowTypeError

from assistant.memory.graphrag_v1.index.typing import ErrorHandlerFn
from assistant.memory.graphrag_v1.index.storage import PipelineStorage
from assistant.memory.graphrag_v1.index.emit.table_emitter import TableEmitter

log = logging.getLogger(__name__)


class ParquetTableEmitter(TableEmitter):
    _storage: PipelineStorage
    _on_error: ErrorHandlerFn

    def __init__(
            self,
            storage: PipelineStorage,
            on_error: ErrorHandlerFn,
    ):
        self._storage = storage
        self._on_error = on_error

    async def emit(
            self,
            name: str,
            data: pd.DataFrame
    ) -> None:
        filename = f"{name}.parquet"
        log.info(f"emitting parquet table {filename}")
        try:
            await self._storage.set(filename, data.to_parquet())
        except ArrowTypeError as e:
            log.exception("Error while emitting parquet table.")
            self._on_error(
                e,
                traceback.format_exc(),
                None,
            )
        except ArrowInvalid as e:
            log.exception("Error while emitting parquet table.")
            self._on_error(
                e,
                traceback.format_exc(),
                None
            )
