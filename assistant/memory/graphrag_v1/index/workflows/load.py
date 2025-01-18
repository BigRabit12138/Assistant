from __future__ import annotations

import logging

from collections.abc import Callable
from typing import (
    Any,
    cast,
    NamedTuple,
    TYPE_CHECKING,
)

from datashaper import Workflow

from assistant.memory.graphrag_v1.index.utils import topological_sort
from assistant.memory.graphrag_v1.index.errors import (
    UnknownWorkflowError,
    UndefinedWorkflowError,
    NoWorkflowsDefinedError,
)
from assistant.memory.graphrag_v1.index.workflows.typing import (
    WorkflowToRun,
    VerbDefinitions,
    WorkflowDefinitions,
)
from assistant.memory.graphrag_v1.index.workflows.default_workflows import (
    default_workflow as _default_workflows
)

if TYPE_CHECKING:
    from assistant.memory.graphrag_v1.index.config import (
        PipelineWorkflowStep,
        PipelineWorkflowConfig,
        PipelineWorkflowReference,
    )

anonymous_workflow_count = 0

VerbFn = Callable[..., Any]
log = logging.getLogger(__name__)


class LoadWorkflowResult(NamedTuple):
    workflows: list[WorkflowToRun]
    dependencies: dict[str, list[str]]


def load_workflows(
        workflows_to_load: list[PipelineWorkflowReference],
        additional_verbs: VerbDefinitions | None = None,
        additional_workflows: WorkflowDefinitions | None = None,
        memory_profile: bool = False,
) -> LoadWorkflowResult:
    workflow_graph: dict[str, WorkflowToRun] = {}

    global anonymous_workflow_count
    for reference in workflows_to_load:
        name = reference.name
        is_anonymous = name is None or name.strip() == ""
        if is_anonymous:
            name = f"Anonymous Workflow {anonymous_workflow_count}"
            anonymous_workflow_count += 1

        name = cast(str, name)

        config = reference.config
        workflow = create_workflow(
            name or "MISSING NAME!",
            reference.steps,
            config,
            additional_verbs,
            additional_workflows,
        )
        workflow_graph[name] = WorkflowToRun(workflow, config=config or {})

    for name in list(workflow_graph.keys()):
        workflow = workflow_graph[name]
        deps = [
            d.replace("workflow:", "")
            for d in workflow.workflow.dependencies
            if d.startswith("workflow:")
        ]
        for dependency in deps:
            if dependency not in workflow_graph:
                reference = {"name": dependency, **workflow.config}
                workflow_graph[dependency] = WorkflowToRun(
                    workflow=create_workflow(
                        dependency,
                        config=reference,
                        additional_verbs=additional_verbs,
                        additional_workflows=additional_workflows,
                        memory_profile=memory_profile,
                    ),
                    config=reference,
                )

    def filter_wf_dependencies(name_: str) -> list[str]:
        externals = [
            e.replace("workflow:", "")
            for e in workflow_graph[name_].workflow.dependencies
        ]
        return [
            e
            for e in externals
            if e in workflow_graph
        ]

    task_graph = {name: filter_wf_dependencies(name) for name in workflow_graph}
    workflow_run_order = topological_sort(task_graph)
    workflows = [workflow_graph[name] for name in workflow_run_order]
    log.info(f"Workflow Run Order: {workflow_run_order}.")
    return LoadWorkflowResult(workflows=workflows, dependencies=task_graph)


def create_workflow(
        name: str,
        steps: list[PipelineWorkflowStep] | None = None,
        config: PipelineWorkflowConfig | None = None,
        additional_verbs: VerbDefinitions | None = None,
        additional_workflows: WorkflowDefinitions | None = None,
        memory_profile: bool = False,
) -> Workflow:
    additional_workflows = {
        **_default_workflows,
        **(additional_workflows or {}),
    }
    steps = steps or _get_steps_for_workflow(
        name,
        config,
        additional_workflows
    )
    steps = _remove_disabled_steps(steps)
    return Workflow(
        verbs=additional_verbs or {},
        schema={
            "name": name,
            "steps": steps,
        },
        validate=False,
        memory_profile=memory_profile,
    )


def _get_steps_for_workflow(
        name: str | None,
        config: PipelineWorkflowConfig | None,
        workflows: dict[str, Callable] | None,
) -> list[PipelineWorkflowStep]:
    if config is not None and "steps" in config:
        return config["steps"]

    if workflows is None:
        raise NoWorkflowsDefinedError

    if name is None:
        raise UndefinedWorkflowError

    if name not in workflows:
        raise UnknownWorkflowError(name)

    return workflows[name](config or {})


def _remove_disabled_steps(
        steps: list[PipelineWorkflowStep],
) -> list[PipelineWorkflowStep]:
    return [
        step
        for step in steps
        if step.get("enabled", True)
    ]
