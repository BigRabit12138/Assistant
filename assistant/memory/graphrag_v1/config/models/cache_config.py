from pydantic import BaseModel, Field

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.config.enums import CacheType


class CacheConfig(BaseModel):
    type: CacheType = Field(
        description='The cache type to use.',
        default=defaults.CACHE_TYPE
    )
    base_dir: str = Field(
        description='The base directory for the cache.',
        default=defaults.CACHE_BASE_DIR
    )
    connection_string: str | None = Field(
        description='The cache connection string to use.',
        default=None
    )
    container_name: str | None = Field(
        description='The cache container name to use.',
        default=None
    )
    storage_account_blob_url: str | None = Field(
        description='The storage account blob url to use.',
        default=None
    )

