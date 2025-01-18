from pydantic import BaseModel, Field

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.config.enums import StorageType


class StorageConfig(BaseModel):
    type: StorageType = Field(
        description="The storage type to use.",
        default=defaults.STORAGE_TYPE,
    )
    base_dir: str = Field(
        description="The base directory for the storage.",
        default=defaults.STORAGE_BASE_DIR,
    )
    connection_string: str | None = Field(
        description="The storage connection string to use.",
        default=None,
    )
    container_name: str | None = Field(
        description="The storage container name to use.",
        default=None,
    )
    storage_account_blob_url: str | None = Field(
        description="The storage account blob url to use.",
        default=None,
    )
