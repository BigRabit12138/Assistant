import json

from typing import Any
from pathlib import Path
from datetime import datetime, timezone

from datashaper import NoopWorkflowCallbacks
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential


class BlobWorkflowCallbacks(NoopWorkflowCallbacks):
    _blob_service_client: BlobServiceClient
    _container_name: str
    _max_block_count: int = 25000

    def __init__(
            self,
            connection_string: str | None,
            container_name: str,
            blob_name: str = "",
            base_dir: str | None = None,
            storage_account_blob_url: str | None = None,
    ):
        if container_name is None:
            msg = "No container name provided for blob storage."
            raise ValueError(msg)

        if connection_string is None and storage_account_blob_url is None:
            msg = "No storage account blob url provided for blob storage."
            raise ValueError(msg)

        self._connection_string = connection_string
        self._storage_account_blob_url = storage_account_blob_url
        if self._connection_string:
            self._blob_service_client = BlobServiceClient.from_connection_string(
                self._connection_string
            )
        else:
            if storage_account_blob_url is None:
                msg = ("Either connection_sting or storage_account_blob_url "
                       "must be provided.")
                raise ValueError(msg)

            self._blob_service_client = BlobServiceClient(
                storage_account_blob_url,
                credential=DefaultAzureCredential()
            )

        if blob_name == "":
            blob_name = f"report/{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d-%H:%M:%S:%f')}.logs.json"

        self._blob_name = str(Path(base_dir or "") / blob_name)
        self._container_name = container_name
        self._blob_client = self._blob_service_client.get_blob_client(
            self._container_name, self._blob_name
        )
        if not self._blob_client.exists():
            self._blob_client.create_append_blob()

        self._num_blocks = 0

    def _write_log(
            self,
            log: dict[str, Any]
    ):
        if (
            self._num_blocks >= self._max_block_count
        ):
            self.__init__(
                self._connection_string,
                self._container_name,
                storage_account_blob_url=self._storage_account_blob_url
            )

        blob_client = self._blob_service_client.get_blob_client(
            self._container_name, self._blob_name
        )
        blob_client.append_block(json.dumps(log) + "\n")

        self._num_blocks += 1

    def on_error(
        self,
        message: str,
        cause: BaseException | None = None,
        stack: str | None = None,
        details: dict | None = None,
    ) -> None:
        self._write_log(
            {
                "type": "error",
                "data": message,
                "cause": str(cause),
                "stack": stack,
                "details": details
            }
        )

    def on_warning(
            self,
            message: str,
            details: dict | None = None
    ) -> None:
        self._write_log(
            {
                "type": "warning",
                "data": message,
                "details": details
            }
        )

    def on_log(
            self,
            message: str,
            details: dict | None = None
    ) -> None:
        self._write_log(
            {
                "type": "log",
                "data": message,
                "details": details
            }
        )
