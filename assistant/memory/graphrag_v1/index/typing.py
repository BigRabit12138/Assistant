from dataclasses import dataclass
from collections.abc import Callable

import pandas as pd

# 错误回调函数
ErrorHandlerFn = Callable[[BaseException | None, str | None, dict | None], None]


@dataclass
class PipelineRunResult:
    workflow: str
    result: pd.DataFrame | None
    errors: list[BaseException] | None
