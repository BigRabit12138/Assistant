from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel
from pydantic import Field as pydantic_Field

from assistant.memory.graphrag_v1.config.enums import ReportingType

T = TypeVar("T")


class PipelineReportingConfig(BaseModel, Generic[T]):
    type: T


class PipelineFileReportingConfig(
    PipelineReportingConfig[Literal[ReportingType.file]]
):
    type: Literal[ReportingType.file] = ReportingType.file

    base_dir: str | None = pydantic_Field(
        description="The base directory for the reporting.",
        default=None
    )


class PipelineConsoleReportingConfig(
    PipelineReportingConfig[Literal[ReportingType.console]]
):
    type: Literal[ReportingType.console] = ReportingType.console


class PipelineBlobReportingConfig(
    PipelineReportingConfig[Literal[ReportingType.blob]]
):
    type: Literal[ReportingType.blob] = ReportingType.blob

    connection_string: str | None = pydantic_Field(
        description="The blob reporting connection string for the reporting.",
        default=None
    )

    container_name: str = pydantic_Field(
        description="The container name for reporting.",
        default=None
    )

    storage_account_blob_url: str | None = pydantic_Field(
        description="The storage account blob url for reporting.",
        default=None
    )

    base_dir: str | None = pydantic_Field(
        description="The base directory for the reporting.",
        default=None
    )


PipelineReportingConfigTypes = (
    PipelineFileReportingConfig
    | PipelineBlobReportingConfig
    | PipelineConsoleReportingConfig
)
