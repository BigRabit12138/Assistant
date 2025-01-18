import logging

from typing import cast
from pathlib import Path
from collections.abc import Awaitable, Callable

import pandas as pd

from assistant.memory.graphrag_v1.config import InputConfig, InputType
from assistant.memory.graphrag_v1.index.config import PipelineInputConfig
from assistant.memory.graphrag_v1.index.progress import (
    NullProgressReporter,
    ProgressReporter
)
from assistant.memory.graphrag_v1.index.storage import (
    BlobPipelineStorage,
    FilePipelineStorage
)

from assistant.memory.graphrag_v1.index.input.csv import load as load_csv
from assistant.memory.graphrag_v1.index.input.csv import input_type as csv
from assistant.memory.graphrag_v1.index.input.text import load as load_text
from assistant.memory.graphrag_v1.index.input.text import input_type as text

log = logging.getLogger(__name__)

loaders: dict[str, Callable[..., Awaitable[pd.DataFrame]]] = {
    text: load_text,
    csv: load_csv,
}


async def load_input(
        config: PipelineInputConfig | InputConfig,
        progress_reporter: ProgressReporter | None = None,
        root_dir: str | None = None,
) -> pd.DataFrame:
    root_dir = root_dir or ""
    log.info(f"loading input from root_dir={config.base_dir}")
    progress_reporter = progress_reporter or NullProgressReporter()

    if config is None:
        msg = "No input specified!"
        raise ValueError(msg)

    match config.type:
        case InputType.blob:
            log.info("using blob storage input.")
            if config.container_name is None:
                msg = "Container name required for blob storage."
                raise ValueError(msg)

            if (
                config.connection_string is None
                and config.storage_account_blob_url is None
            ):
                msg = ("Connection string or storage account blob "
                       "url required for blob storage.")
                raise ValueError(msg)

            storage = BlobPipelineStorage(
                connection_string=config.connection_string,
                storage_account_blob_url=config.storage_account_blob_url,
                container_name=config.container_name,
                path_prefix=config.base_dir
            )
        case InputType.file:
            log.info("using file storage for input.")
            storage = FilePipelineStorage(
                root_dir=str(Path(root_dir) / (config.base_dir or ""))
            )
        case _:
            log.info("using file storage for input.")
            storage = FilePipelineStorage(
                root_dir=str(Path(root_dir) / (config.base_dir or ""))
            )

    if config.file_type in loaders:
        progress = progress_reporter.child(
            f"Loading Input ({config.file_type})",
            transient=False
        )
        loader = loaders[config.file_type]
        results = await loader(config, progress, storage)
        return cast(pd.DataFrame, results)

    msg = f"Unknown input type {config.file_type}."
    raise ValueError(msg)
