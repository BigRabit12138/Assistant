from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel
from pydantic import Field as pydantic_Field

from assistant.memory.graphrag_v1.config.enums import StorageType

T = TypeVar("T")


class PipelineStorageConfig(BaseModel, Generic[T]):
    type: T


class PipelineFileStorageConfig(PipelineStorageConfig[Literal[StorageType.file]]):
    type: Literal[StorageType.file] = StorageType.file

    base_dir: str | None = pydantic_Field(
        description="The base directory for the storage.",
        default=None
    )


class PipelineMemoryStorageConfig(PipelineStorageConfig[Literal[StorageType.memory]]):
    type: Literal[StorageType.memory] = StorageType.memory


class PipelineBlobStorageConfig(PipelineStorageConfig[Literal[StorageType.memory]]):
    type: Literal[StorageType.blob] = StorageType.blob

    connection_string: str | None = pydantic_Field(
        description="The blob storage connection string for the storage.",
        default=None
    )

    container_name: str = pydantic_Field(
        description="The container name for storage",
        default=None
    )

    base_dir: str | None = pydantic_Field(
        description="The base directory for the storage.",
        default=None
    )

    storage_account_blob_url: str | None = pydantic_Field(
        description="The storage account blob url.",
        default=None
    )


PipelineStorageConfigTypes = (
    PipelineFileStorageConfig | PipelineMemoryStorageConfig | PipelineBlobStorageConfig
)
