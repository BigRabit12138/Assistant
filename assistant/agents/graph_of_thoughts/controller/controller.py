import json
import logging

from typing import List

from assistant.agents.graph_of_thoughts.parser import Parser
from assistant.agents.prompts.prompter import Prompter
from assistant.agents.graph_of_thoughts.operations import GraphOfOperations, Thought
from assistant.llm.language_models import AbstractLanguageModel


class Controller:
    def __init__(
            self,
            lm: AbstractLanguageModel,
            graph: GraphOfOperations,
            prompter: Prompter,
            parser: Parser,
            problem_parameter: dict
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__module__)
        self.lm = lm
        self.graph = graph
        self.prompter = prompter
        self.parser = parser
        self.problem_parameters = problem_parameter
        self.run_executed = False

    def run(self) -> None:
        self.logger.debug("Checking that the program is in a valid state")
        assert self.graph.roots is not None, "The operations graph has no root"
        self.logger.debug("The program is in a valid state")

        execution_queue = [
            operation
            for operation in self.graph.operations
            if operation.can_be_executed()
        ]

        while len(execution_queue) > 0:
            current_operation = execution_queue.pop(0)
            self.logger.info(f"Executing operation {current_operation.operation_type}")
            current_operation.execute(
                self.lm, self.prompter, self.parser, **self.problem_parameters
            )
            self.logger.info(f"Operation {current_operation.operation_type} executed")
            for operation in current_operation.successors:
                assert (
                    operation in self.graph.operations
                ), "The successor of an operation is not in the operations graph"
                if operation.can_be_executed():
                    execution_queue.append(operation)

        self.logger.info("All operation executed")
        self.run_executed = True

    def get_final_thoughts(self) -> List[List[Thought]]:
        assert self.run_executed, "The run method has not been executed"
        return [operation.get_thoughts() for operation in self.graph.leaves]

    def output_graph(self, path: str) -> None:
        output = []
        for operation in self.graph.operations:
            operation_serialized = {
                "operation": operation.operation_type.name,
                'thoughts': [thought.state for thought in operation.get_thoughts()],
            }
            if any([thought.scored for thought in operation.get_thoughts()]):
                operation_serialized['scored'] = [
                    thought.scored for thought in operation.get_thoughts()
                ]
                operation_serialized['scores'] = [
                    thought.score for thought in operation.get_thoughts()
                ]

            if any([thought.validated for thought in operation.get_thoughts()]):
                operation_serialized['validated'] = [
                    thought.validated for thought in operation.get_thoughts()
                ]
                operation_serialized['validity'] = [
                    thought.valid for thought in operation.get_thoughts()
                ]

            if any(
                    [
                        thought.compared_to_ground_truth
                        for thought in operation.get_thoughts()
                    ]
            ):
                operation_serialized['compared_to_ground_truth'] = [
                    thought.compared_to_ground_truth
                    for thought in operation.get_thoughts()
                ]
                operation_serialized['problem_solved'] = [
                    thought.solved for thought in operation.get_thoughts()
                ]
            output.append(operation_serialized)
        output.append(
            {
                'prompt_tokens': self.lm.prompt_tokens,
                'completion_tokens': self.lm.completion_tokens,
                'cost': self.lm.cost,
            }
        )

        with open(path, 'w') as file:
            file.write(json.dumps(output, indent=2))
