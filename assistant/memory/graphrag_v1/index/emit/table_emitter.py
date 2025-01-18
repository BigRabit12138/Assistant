from typing import Protocol

import pandas as pd


class TableEmitter(Protocol):
    async def emit(
            self,
            name: str,
            data: pd.DataFrame
    ) -> None:
        pass
