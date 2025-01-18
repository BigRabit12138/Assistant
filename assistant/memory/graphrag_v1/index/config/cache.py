from __future__ import annotations

from typing import (
    Generic,
    Literal,
    TypeVar,
)

from pydantic import BaseModel
from pydantic import Field as pydantic_Field

from assistant.memory.graphrag_v1.config.enums import CacheType

T = TypeVar("T")


class PipelineCacheConfig(BaseModel, Generic[T]):
    type: T


class PipelineFileCacheConfig(PipelineCacheConfig[Literal[CacheType.file]]):
    type: Literal[CacheType.file] = CacheType.file

    base_dir: str | None = pydantic_Field(
        description="The base directory for the cache.",
        default=None
    )


class PipelineMemoryCacheConfig(PipelineCacheConfig[Literal[CacheType.memory]]):
    type: Literal[CacheType.memory] = CacheType.memory


class PipelineNoneCacheConfig(PipelineCacheConfig[Literal[CacheType.none]]):
    type: Literal[CacheType.none] = CacheType.none


class PipelineBlobCacheConfig(PipelineCacheConfig[Literal[CacheType.blob]]):
    type: Literal[CacheType.blob] = CacheType.blob

    base_dir: str | None = pydantic_Field(
        description="The base directory for the cache.",
        default=None
    )

    connection_string: str | None = pydantic_Field(
        description="The blob cache connection string for the cache.",
        default=None
    )

    container_name: str = pydantic_Field(
        description="The container name for cache.",
        default=None
    )

    storage_account_blob_url: str | None = pydantic_Field(
        description="The storage account blob url for cache",
        default=None
    )


PipelineCacheConfigTypes = (
    PipelineFileCacheConfig
    | PipelineBlobCacheConfig
    | PipelineNoneCacheConfig
    | PipelineMemoryCacheConfig
)
