from typing import Any
from collections.abc import Callable
from dataclasses import dataclass as dc_dataclass

from datashaper import (
    Workflow,
    TableContainer,
)

StepDefinition = dict[str, Any]

VerbDefinitions = dict[str, Callable[..., TableContainer]]

WorkflowConfig = dict[str, Any]

WorkflowDefinitions = dict[str, Callable[[WorkflowConfig], list[StepDefinition]]]

VerbTiming = dict[str, float]


@dc_dataclass
class WorkflowToRun:
    workflow: Workflow
    config: dict[str, Any]
