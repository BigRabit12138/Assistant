from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field as pydantic_Field

PipelineWorkflowStep = dict[str, Any]

PipelineWorkflowConfig = dict[str, Any]


class PipelineWorkflowReference(BaseModel):
    name: str | None = pydantic_Field(
        description="Name of the workflow.",
        default=None
    )

    steps: list[PipelineWorkflowStep] | None = pydantic_Field(
        description="The optional steps for the workflow.",
        default=None
    )

    config: PipelineWorkflowConfig | None = pydantic_Field(
        description="The optional configuration for the workflow.",
        default=None
    )
