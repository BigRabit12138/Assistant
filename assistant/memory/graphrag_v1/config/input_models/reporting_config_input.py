from typing_extensions import NotRequired, TypedDict

from assistant.memory.graphrag_v1.config.enums import ReportingType


class ReportingConfigInput(TypedDict):
    type: NotRequired[ReportingType | str | None]
    base_dir: NotRequired[str | None]
    connection_string: NotRequired[str | None]
    container_name: NotRequired[str | None]
    storage_account_blob_url: NotRequired[str | None]
