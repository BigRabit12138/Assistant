from pydantic import BaseModel, Field

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.config.enums import ReportingType


class ReportingConfig(BaseModel):
    type: ReportingType = Field(
        description="The reporting type to use.",
        default=defaults.REPORTING_TYPE,
    )
    base_dir: str = Field(
        description="The base directory for reporting.",
        default=defaults.REPORTING_BASE_DIR,
    )
    connection_string: str | None = Field(
        description="The reporting connection string to use.",
        default=None
    )
    container_name: str | None = Field(
        description="The reporting container name to use.",
        default=None,
    )
    storage_account_blob_url: str | None = Field(
        description="The storage account blob url to use.",
        default=None,
    )
