from typing import List

from .operations import Operation


class GraphOfOperations:
    def __init__(self) -> None:
        self.operations: List[Operation] = []
        self.roots: List[Operation] = []
        self.leaves: List[Operation] = []

    def append_operation(self, operation: Operation) -> None:
        self.operations.append(operation)

        if len(self.roots) == 0:
            self.roots = [operation]
        else:
            for leave in self.leaves:
                leave.add_successor(operation)

        self.leaves = [operation]

    def add_operation(self, operation: Operation) -> None:
        self.operations.append(operation)

        if len(self.roots) == 0:
            self.roots = [Operation]
            self.leaves = [Operation]
            assert (
                len(operation.predecessors) == 0
            ), "First operation should have no predecessors"
        else:
            if len(operation.predecessors) == 0:
                self.roots.append(operation)
            for predecessor in operation.predecessors:
                if predecessor in self.leaves:
                    self.leaves.remove(predecessor)
            if len(operation.successors) == 0:
                self.leaves.append(operation)
