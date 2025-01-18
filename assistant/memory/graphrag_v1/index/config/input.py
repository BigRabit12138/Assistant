from __future__ import annotations

from typing import (
    Generic,
    Literal,
    TypeVar,
)

from pydantic import BaseModel
from pydantic import Field as pydantic_Field

from assistant.memory.graphrag_v1.config.enums import (
    InputType,
    InputFileType
)
from assistant.memory.graphrag_v1.index.config.workflow import (
    PipelineWorkflowStep
)

T = TypeVar("T")


class PipelineInputConfig(BaseModel, Generic[T]):
    file_type: T

    type: InputType | None = pydantic_Field(
        description="The input type to use.",
        default=None
    )

    connection_string: str | None = pydantic_Field(
        description="The blob cache connection string for the input files.",
        default=None
    )

    storage_account_blob_url: str | None = pydantic_Field(
        description="The storage account blob url for the input files.",
        default=None
    )

    container_name: str | None = pydantic_Field(
        description="The container name for input files.",
        default=None
    )

    base_dir: str | None = pydantic_Field(
        description="The base directory for the input files.",
        default=None
    )

    file_pattern: str = pydantic_Field(
        description="The regex file pattern for the input files.",
    )

    file_filter: dict[str, str] | None = pydantic_Field(
        description="The optional file filter for the input files.",
        default=None
    )

    post_process: list[PipelineWorkflowStep] | None = pydantic_Field(
        description="The post processing steps for the input.",
        default=None
    )

    encoding: str | None = pydantic_Field(
        description="The encoding for the input files.",
        default=None
    )


class PipelineCSVInputConfig(
    PipelineInputConfig[Literal[InputFileType.csv]]
):
    file_type: Literal[InputFileType.csv] = InputFileType.csv

    source_column: str | None = pydantic_Field(
        description="The column to use as the source of the document.",
        default=None
    )

    timestamp_column: str | None = pydantic_Field(
        description="The column to use as the timestamp of the document.",
        default=None
    )

    timestamp_format: str | None = pydantic_Field(
        description="The format of the timestamp column, so it can be parsed correctly.",
        default=None,
    )

    text_column: str | None = pydantic_Field(
        description="The column to use as the text of the document.",
        default=None
    )

    title_column: str | None = pydantic_Field(
        description="The column to use as the title of the document.",
        default=None
    )


class PipelineTextInputConfig(
    PipelineInputConfig[Literal[InputFileType.text]]
):
    file_type: Literal[InputFileType.text] = InputFileType.text

    title_text_length: int | None = pydantic_Field(
        description="Number of characters to use from the text as the title.",
        default=None
    )


PipelineInputConfigTypes = PipelineCSVInputConfig | PipelineTextInputConfig
