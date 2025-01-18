from __future__ import annotations

from typing import cast

from assistant.memory.graphrag_v1.config import StorageType
from assistant.memory.graphrag_v1.index.config.storage import (
    PipelineStorageConfig,
    PipelineFileStorageConfig,
    PipelineBlobStorageConfig
)
from assistant.memory.graphrag_v1.index.storage.file_pipline_storage import create_file_storage
from assistant.memory.graphrag_v1.index.storage.blob_pipeline_storage import create_blob_storage
from assistant.memory.graphrag_v1.index.storage.memory_pipeline_storage import create_memory_storage


def load_storage(config: PipelineStorageConfig):
    match config.type:
        case StorageType.memory:
            return create_memory_storage()
        case StorageType.blob:
            config = cast(PipelineBlobStorageConfig, config)
            return create_blob_storage(
                config.connection_string,
                config.storage_account_blob_url,
                config.container_name,
                config.base_dir
            )
        case StorageType.file:
            config = cast(PipelineFileStorageConfig, config)
            return create_file_storage(config.base_dir)
        case _:
            msg = f"Unknown storage type: {config.type}"
            raise ValueError(msg)
