from __future__ import annotations

from devtools import pformat
from pydantic import BaseModel
from pydantic import Field as pydantic_Field

from assistant.memory.graphrag_v1.index.config.cache import (
    PipelineCacheConfigTypes
)
from assistant.memory.graphrag_v1.index.config.input import (
    PipelineInputConfigTypes
)
from assistant.memory.graphrag_v1.index.config.storage import (
    PipelineStorageConfigTypes
)
from assistant.memory.graphrag_v1.index.config.workflow import (
    PipelineWorkflowReference
)
from assistant.memory.graphrag_v1.index.config.reporting import (
    PipelineReportingConfigTypes
)


class PipelineConfig(BaseModel):
    def __repr__(self) -> str:
        return pformat(self, highlight=False)

    def __str__(self):
        return str(self.model_dump_json(indent=4))

    extends: list[str] | str | None = pydantic_Field(
        description="Extends another pipeline configuration.",
        default=None
    )

    input: PipelineInputConfigTypes | None = pydantic_Field(
        default=None,
        discriminator="file_type"
    )

    reporting: PipelineReportingConfigTypes | None = pydantic_Field(
        default=None,
        discriminator="type"
    )

    storage: PipelineStorageConfigTypes | None = pydantic_Field(
        default=None,
        discriminator="type"
    )

    cache: PipelineCacheConfigTypes | None = pydantic_Field(
        default=None,
        discriminator="type"
    )

    root_dir: str | None = pydantic_Field(
        description="The root directory for the pipeline. All other \
        paths will be based on this root_dir.",
        default=None
    )

    workflows: list[PipelineWorkflowReference] = pydantic_Field(
        description="The workflows for the pipeline.",
        default_factory=list
    )
