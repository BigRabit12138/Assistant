from __future__ import annotations

from typing import TYPE_CHECKING, cast

from assistant.memory.graphrag_v1.config.enums import CacheType
from assistant.memory.graphrag_v1.index.config.cache import (
    PipelineBlobCacheConfig,
    PipelineFileCacheConfig
)
from assistant.memory.graphrag_v1.index.storage import (
    BlobPipelineStorage,
    FilePipelineStorage
)

if TYPE_CHECKING:
    from assistant.memory.graphrag_v1.index.config import PipelineCacheConfig

from assistant.memory.graphrag_v1.index.cache.json_pipeline_cache import JsonPipelineCache
from assistant.memory.graphrag_v1.index.cache.noop_pipeline_cache import NoopPipelineCache
from assistant.memory.graphrag_v1.index.cache.memory_pipeline_cache import create_memory_cache


def load_cache(
        config: PipelineCacheConfig | None,
        root_dir: str | None
):
    if config is None:
        return NoopPipelineCache()

    match config.type:
        case CacheType.none:
            return NoopPipelineCache()
        case CacheType.memory:
            return create_memory_cache()
        case CacheType.file:
            config = cast(PipelineFileCacheConfig, config)
            storage = FilePipelineStorage(root_dir).child(config.base_dir)
            return JsonPipelineCache(storage)
        case CacheType.blob:
            config = cast(PipelineBlobCacheConfig, config)
            storage = BlobPipelineStorage(
                config.connection_string,
                config.container_name,
                storage_account_blob_url=config.storage_account_blob_url
            ).child(config.base_dir)
            return JsonPipelineCache(storage)
        case _:
            msg = f"Unknown cache type: {config.type}"
            raise ValueError(msg)
