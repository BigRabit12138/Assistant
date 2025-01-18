import json
import logging

from pathlib import Path
from io import TextIOWrapper

from datashaper import NoopWorkflowCallbacks

log = logging.getLogger(__name__)


class FileWorkflowCallbacks(NoopWorkflowCallbacks):
    _out_stream: TextIOWrapper

    def __init__(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._out_stream = open(
            Path(directory) / "logs.json", "a", encoding="utf-8"
        )

    def on_error(
        self,
        message: str,
        cause: BaseException | None = None,
        stack: str | None = None,
        details: dict | None = None,
    ) -> None:
        self._out_stream.write(
            json.dumps(
                {
                    "type": "error",
                    "data": message,
                    "stack": stack,
                    "source": str(cause),
                    "details": details
                }
            )
            + "\n"
        )
        message = f"{message} details={details}."
        log.info(message)

    def on_warning(
            self,
            message: str,
            details: dict | None = None
    ) -> None:
        self._out_stream.write(
            json.dumps(
                {
                    "type": "warning",
                    "data": message,
                    "details": details
                }
            )
            + "\n"
        )
        _print_warning(message)

    def on_log(
            self,
            message: str,
            details: dict | None = None
    ) -> None:
        self._out_stream.write(
            json.dumps(
                {
                    "type": "log",
                    "data": message,
                    "details": details
                }
            )
            + "\n"
        )
        message = f"{message} details={details}."
        log.info(message)


def _print_warning(skk):
    log.warning(skk)
