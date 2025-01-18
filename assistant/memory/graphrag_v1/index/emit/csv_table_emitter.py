import logging

import pandas as pd

from assistant.memory.graphrag_v1.index.storage import PipelineStorage
from assistant.memory.graphrag_v1.index.emit.table_emitter import TableEmitter

log = logging.getLogger(__name__)


class CSVTableEmitter(TableEmitter):
    _storage: PipelineStorage

    def __init__(
            self,
            storage: PipelineStorage
    ):
        self._storage = storage

    async def emit(
            self,
            name: str,
            data: pd.DataFrame
    ) -> None:
        filename = f"{name}.csv"
        log.info(f"emitting CSV table {filename}")
        await self._storage.set(
            filename,
            data.to_csv()
        )
