from typing import Any

from datashaper import (
    Progress,
    ExecutionNode,
    TableContainer,
    NoopWorkflowCallbacks
)

from assistant.memory.graphrag_v1.index.progress import ProgressReporter


class ProgressWorkflowCallbacks(NoopWorkflowCallbacks):
    _root_progress: ProgressReporter
    _progress_stack: list[ProgressReporter]

    def __init__(
            self,
            progress: ProgressReporter
    ) -> None:
        self._progress = progress
        self._progress_stack = [progress]

    def _pop(self) -> None:
        self._progress_stack.pop()

    def _push(self, name: str) -> None:
        self._progress_stack.append(self._latest.child(name))

    @property
    def _latest(self) -> ProgressReporter:
        return self._progress_stack[-1]

    def on_workflow_start(
            self,
            name: str,
            instance: object
    ) -> None:
        self._push(name)

    def on_workflow_end(
            self,
            name: str,
            instance: object
    ) -> None:
        self._pop()

    def on_step_start(
            self,
            node: ExecutionNode,
            inputs: dict
    ) -> None:
        verb_id_str = f"({node.node_id})" if node.has_explicit_id else ""
        self._push(f"Verb {node.verb.name}{verb_id_str}")
        self._latest(Progress(percent=0))

    def on_step_end(
            self,
            node: ExecutionNode,
            result: TableContainer | None
    ) -> None:
        self._pop()

    def on_step_progress(
            self,
            node: ExecutionNode,
            progress: Progress
    ) -> None:
        self._latest(progress)
