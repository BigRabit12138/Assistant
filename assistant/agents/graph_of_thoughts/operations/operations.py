import logging
import itertools

from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Callable, Union

from assistant.agents.graph_of_thoughts.parser import Parser
from assistant.agents.prompts.prompter import Prompter
from assistant.agents.graph_of_thoughts.operations.thought import Thought
from assistant.llm.language_models import AbstractLanguageModel


class OperationType(Enum):
    score: int = 0
    validate_and_improve: int = 1
    generate: int = 2
    improve: int = 3
    aggregate: int = 4
    keep_best_n: int = 5
    keep_valid: int = 6
    ground_truth_evaluator: int = 7
    selector: int = 8


class Operation(ABC):
    _ids: Iterable[int] = itertools.count(0)

    operation_type: OperationType = None

    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Operation._ids)
        self.predecessors: List[Operation] = []
        self.successors: List[Operation] = []
        self.executed: bool = False

    def can_be_executed(self) -> bool:
        return all(predecessor.executed for predecessor in self.predecessors)

    def get_previous_thought(self) -> List[Thought]:
        previous_thoughts: List[Thought] = [
            thought
            for predecessor in self.predecessors
            for thought in predecessor.get_thoughts()
        ]

        return previous_thoughts

    def add_predecessor(
            self,
            operation: Operation
    ) -> None:
        self.predecessors.append(operation)
        operation.successors.append(self)

    def add_successor(
            self,
            operation: Operation
    ) -> None:
        self.successors.append(operation)
        operation.predecessors.append(self)

    def execute(
            self,
            lm: AbstractLanguageModel,
            prompter: Prompter,
            parser: Parser,
            **kwargs
    ) -> None:
        assert self.can_be_executed(), "Not all predecessors have been executed"
        self.logger.info(
            f"Executing operation {self.id} of type {self.operation_type}"
        )
        self._execute(lm, prompter, parser, **kwargs)
        self.logger.debug(f"Operation {self.id} executed")
        self.executed = True

    @abstractmethod
    def _execute(
            self,
            lm: AbstractLanguageModel,
            prompter: Prompter,
            parser: Parser,
            **kwargs
    ) -> None:
        pass

    @abstractmethod
    def get_thoughts(self) -> List[Thought]:
        pass


class Score(Operation):
    operation_type: OperationType = OperationType.score

    def __init__(
            self,
            num_samples: int = 1,
            combined_scoring: bool = False,
            scoring_function: Callable[
                [Union[List[Dict], Dict]], Union[List[float], float]
            ] = None,
    ) -> None:
        super().__init__()
        self.num_samples: int = num_samples
        self.combined_scoring: bool = combined_scoring
        self.thoughts: List[Thought] = []
        self.scoring_function: Callable[
            [Union[List[Dict], Dict]], Union[List[float], float]
        ] = scoring_function

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(
            self,
            lm: AbstractLanguageModel,
            prompter: Prompter,
            parser: Parser,
            **kwargs
    ) -> None:
        previous_thoughts: List[Thought] = self.get_previous_thought()

        assert (
            len(self.predecessors) > 0
        ), "Score operation needs at least one predecessor"

        if self.combined_scoring:
            previous_thoughts_states = [thought.state for thought in previous_thoughts]
            if self.scoring_function is not None:
                self.logger.debug(
                    f"Using scoring function {self.scoring_function} to score states"
                )
                scores = self.scoring_function(previous_thoughts_states)
            else:
                prompt = prompter.score_prompt(previous_thoughts_states)
                self.logger.debug(f"Prompt for LM: {prompt}")

                responses = lm.get_response_texts(
                    lm.query(prompt, num_responses=self.num_samples)
                )
                self.logger.debug(f"Responses from LM: {responses}")
                scores = parser.parse_score_answer(previous_thoughts_states, responses)
            for thought, score in zip(previous_thoughts, scores):
                new_thought = Thought.from_thought(thought)
                new_thought.score = score
                self.thoughts.append(new_thought)
        else:
            for thought in previous_thoughts:
                new_thought = Thought.from_thought(thought)
                if self.scoring_function is not None:
                    self.logger.debug(
                        f"Using scoring function {self.scoring_function} to score state"
                    )
                    score = self.scoring_function(thought.state)
                else:
                    prompt = prompter.score_prompt([thought.state])
                    self.logger.debug(f"Prompt for LM: {prompt}")

                    responses = lm.get_response_texts(
                        lm.query(prompt, num_responses=self.num_samples)
                    )
                    self.logger.debug(f"Response from LM: {responses}")
                    score = parser.parse_score_answer([thought.state], responses)[0]

                new_thought.score = score
                self.thoughts.append(new_thought)

        self.logger.info(
            f"Score operation {self.id} scored {len(self.thoughts)} thoughts"
        )


class ValidateAndImprove(Operation):
    operation_type: OperationType = OperationType.validate_and_improve

    def __init__(
            self,
            num_samples: int = 1,
            improve: bool = True,
            num_tries: int = 3,
            validate_function: Callable[[Dict], bool] = None,
    ) -> None:
        super().__init__()
        self.num_samples: int = num_samples
        self.improve: bool = improve
        self.num_tries: int = num_tries
        self.validate_function: Callable[[Dict], bool] = validate_function
        self.thoughts: List[List[Thought]] = []

    def get_thoughts(self) -> List[Thought]:
        return [thought_list[-1] for thought_list in self.thoughts]

    def _execute(
            self,
            lm: AbstractLanguageModel,
            prompter: Prompter,
            parser: Parser,
            **kwargs
    ) -> None:
        previous_thoughts: List[Thought] = self.get_previous_thought()

        assert (
            len(self.predecessors) > 0
        ), "ValidateAndImprove operation needs at least one predecessor"

        for thought in previous_thoughts:
            thought_list = []
            current_thought = Thought.from_thought(thought)
            current_try = 0
            while True:
                if self.validate_function is not None:
                    self.logger.debug(
                        f"Using validate function {self.validate_function} to score states"
                    )
                    valid = self.validate_function(current_thought.state)
                else:
                    prompt = prompter.validation_prompt(**current_thought.state)
                    self.logger.debug(f"Prompt for LM: {prompt}")
                    responses = lm.get_response_texts(
                        lm.query(prompt, num_responses=self.num_samples)
                    )
                    self.logger.debug(f"Response from LM: {responses}")

                    valid = parser.parse_validation_answer(
                        current_thought.state, responses
                    )

                current_thought.valid = valid
                thought_list.append(current_thought)
                if (
                    not self.improve
                    or current_thought.valid
                    or current_try >= self.num_tries
                ):
                    break

                improve_prompt = prompter.improve_prompt(**current_thought.state)
                self.logger.debug(f"Prompt for LM: {improve_prompt}")
                responses = lm.get_response_texts(
                    lm.query(improve_prompt, num_responses=1)
                )
                self.logger.debug("Responses from LM: %s", responses)
                state_update = parser.parse_improve_answer(
                    current_thought.state, responses
                )
                current_thought = Thought({**current_thought.state, **state_update})
                current_try += 1

            self.thoughts.append(thought_list)

        valid_thoughts_num = len(
            [
                thought_list[-1]
                for thought_list in self.thoughts
                if thought_list[-1].valid
            ]
        )
        self.logger.info(
            f"Validate and improve operation {self.id} created {valid_thoughts_num}"
            f"valid thoughts from {len(previous_thoughts)} previous thoughts"
        )


class Generate(Operation):
    operation_type: OperationType = OperationType.generate

    def __init__(
            self,
            num_branches_prompt: int = 1,
            num_branches_response: int = 1
    ) -> None:
        super().__init__()
        self.num_branches_prompt: int = num_branches_prompt
        self.num_branches_response: int = num_branches_response
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(
            self,
            lm: AbstractLanguageModel,
            prompter: Prompter,
            parser: Parser,
            **kwargs
    ) -> None:
        previous_thoughts: List[Thought] = self.get_previous_thought()

        if len(previous_thoughts) == 0 and len(self.predecessors) > 0:
            return

        if len(previous_thoughts) == 0:
            previous_thoughts = [Thought(state=kwargs)]

        for thought in previous_thoughts:
            base_state = thought.state
            prompt = prompter.generate_prompt(self.num_branches_prompt, **base_state)
            self.logger.debug(f"Prompt for LM: {prompt}")
            responses = lm.get_response_texts(
                lm.query(prompt, num_responses=self.num_branches_response)
            )
            self.logger.debug(f"Responses from LM: {responses}")
            for new_state in parser.parse_generate_answer(base_state, responses):
                new_state = {**base_state, **new_state}
                self.thoughts.append(Thought(new_state))
                self.logger.debug(
                    f"New thought {self.thoughts[-1].id} created with state {self.thoughts[-1].state}"
                )

        if (
            len(self.thoughts)
            > self.num_branches_prompt
            * self.num_branches_response
            * len(previous_thoughts)
            and self.num_branches_prompt > 0
        ):
            self.logger.warning(
                f"Generate operation {self.id} created more thoughts than expected"
            )
        self.logger.info(
            f"Generate operation {self.id} created {self.thoughts} new thoughts"
        )


class Improve(Operation):
    operation_type: OperationType = OperationType.improve

    def __init__(self) -> None:
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(
            self,
            lm: AbstractLanguageModel,
            prompter: Prompter,
            parser: Parser,
            **kwargs
    ) -> None:
        previous_thoughts: List[Thought] = self.get_previous_thought()

        assert len(self.predecessors) > 0, "Needs at least one predecessor"

        for thought in previous_thoughts:
            improve_prompt = prompter.improve_prompt(**thought.state)
            self.logger.debug(f"Prompt for LM: {improve_prompt}")
            responses = lm.get_response_texts(
                lm.query(improve_prompt, num_responses=1)
            )
            self.logger.debug(f"Responses from LM: {responses}")
            state_update = parser.parse_improve_answer(thought.state, responses)
            self.thoughts.append(Thought({**thought.state, **state_update}))

        self.logger.info(
            f"Improve operation {self.id} improved {len(self.thoughts)} thoughts"
        )


class Aggregate(Operation):
    operation_type: OperationType = OperationType.aggregate

    def __init__(
            self,
            num_responses: int = 1
    ) -> None:
        super().__init__()
        self.thoughts: List[Thought] = []
        self.num_responses: int = num_responses

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(
            self,
            lm: AbstractLanguageModel,
            prompter: Prompter,
            parser: Parser,
            **kwargs
    ) -> None:
        assert (
            len(self.predecessors) >= 1
        ), "Aggregate operation must have at least one predecessor"

        previous_thoughts: List[Thought] = self.get_previous_thought()

        if len(previous_thoughts) == 0:
            return

        base_state: Dict = {}
        for thought in sorted(previous_thoughts, key=lambda thought: thought.score):
            base_state = {**base_state, **thought.state}

        previous_thought_states = [thought.state for thought in previous_thoughts]
        prompt = prompter.aggregation_prompt(previous_thought_states)

        self.logger.debug(f"Prompt for LM: {prompt}")

        responses = lm.get_response_texts(
            lm.query(prompt, num_responses=self.num_responses)
        )
        self.logger.debug(f"Responses from LM: {responses}")
        parsed = parser.parse_aggregation_answer(previous_thought_states, responses)

        if isinstance(parsed, dict):
            parsed = [parsed]

        for new_state in parsed:
            self.thoughts.append(Thought({**base_state, **new_state}))


class KeepBestN(Operation):
    operation_type: OperationType = OperationType.keep_best_n

    def __init__(
            self,
            n: int,
            higher_is_better: bool = True
    ) -> None:
        super().__init__()
        self.n: int = n
        assert self.n > 0, "KeepBestN operation must keep at least one thought"
        self.higher_is_better: bool = higher_is_better
        self.thoughts: List[Thought] = []

    def get_best_n(self) -> List[Thought]:
        previous_thoughts: List[Thought] = self.get_previous_thought()

        assert all(
            previous_thought.scored for previous_thought in previous_thoughts
        ), "Not all thoughts have been scored"

        try:
            return sorted(
                previous_thoughts,
                key=lambda thought: thought.score,
                reverse=self.higher_is_better,
            )[: self.n]
        except Exception as e:
            self.logger.error(f"Error {e} in KeepBestN operation")
            self.logger.error(
                f"Previous operation: {[op.id for op in self.predecessors]}"
            )
            self.logger.error(f"Previous thoughts: {previous_thoughts}")
            self.logger.error(
                f"Scores: {[thought.score for thought in previous_thoughts]}"
            )
            return sorted(
                [i for i in previous_thoughts if isinstance(i.score, float)],
                key=lambda thought: thought.score,
                reverse=self.higher_is_better
            )[: self.n]

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(
            self,
            lm: AbstractLanguageModel,
            prompter: Prompter,
            parser: Parser,
            **kwargs
    ) -> None:
        assert (
            len(self.predecessors) >= 1
        ), "KeepBestN operation must have at least one predecessor"

        self.thoughts = [Thought.from_thought(thought) for thought in self.get_best_n()]

        for thought in self.thoughts:
            self.logger.debug(
                f"Thought {thought.id} with state {thought.state} kept"
            )

        self.logger.info(
            f"KeepBestN operation {self.id} kept {len(self.thoughts)} thoughts"
        )


class KeepValid(Operation):
    operation_type: OperationType = OperationType.keep_valid

    def __init__(self) -> None:
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(
            self,
            lm: AbstractLanguageModel,
            prompter: Prompter,
            parser: Parser,
            **kwargs
    ) -> None:
        assert (
            len(self.predecessors) >= 1
        ), "KeepValid operation must have at least one predecessor"

        self.thoughts: List[Thought] = [
            Thought.from_thought(thought)
            for thought in self.get_previous_thought()
            if not thought.validated or thought.valid
        ]

        if any(not thought.validated for thought in self.thoughts):
            self.logger.warning(
                f"KeepValid operation {self.id} has unvalidated thoughts"
            )

        for thought in self.thoughts:
            self.logger.debug(
                f"Thought {thought.id} with state {thought.state} kept"
            )

        self.logger.info(
            f"KeepValid operation {self.id} kept {len(self.thoughts)} thoughts"
        )


class GroundTruth(Operation):
    operation_type: OperationType = OperationType.ground_truth_evaluator

    def __init__(
            self,
            ground_truth_evaluator: Callable[[Dict], bool]
    ) -> None:
        super().__init__()
        self.ground_truth_evaluator: Callable[[Dict], bool] = ground_truth_evaluator
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(
            self,
            lm: AbstractLanguageModel,
            prompter: Prompter,
            parser: Parser,
            **kwargs
    ) -> None:
        assert (
            len(self.predecessors) >= 1
        ), "GroundTruth operation must have at least one predecessor"

        previous_thoughts: List[Thought] = self.get_previous_thought()

        for thought in previous_thoughts:
            new_thought = Thought.from_thought(thought)
            try:
                new_thought.solved = self.ground_truth_evaluator(new_thought.state)
            except Exception as e:
                self.logger.error(f"GroundTruth happens error: {e}")
                new_thought.solved = False

            self.thoughts.append(new_thought)

        self.logger.info(
            f"GroundTruth operation {self.id} evaluated {len(self.thoughts)}"
            f"thoughts and {len([thought for thought in self.thoughts if thought.solved])}"
            "solved the problem"
        )


class Selector(Operation):
    operation_type: OperationType = OperationType.selector

    def __init__(
            self,
            selector: Callable[[List[Thought]], List[Thought]]
    ) -> None:
        super().__init__()
        self.selector: Callable[[List[Thought]], List[Thought]] = selector
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(
            self,
            lm: AbstractLanguageModel,
            prompter: Prompter,
            parser: Parser,
            **kwargs
    ) -> None:
        previous_thoughts: List[Thought] = self.get_previous_thought()

        if len(previous_thoughts) == 0:
            previous_thoughts = [Thought(kwargs)]

        self.thoughts = [
            Thought.from_thought(thought)
            for thought in self.selector(previous_thoughts)
        ]

        for thought in self.thoughts:
            self.logger.debug(
                f"Thought {thought.id} with state {thought.state} selected"
            )

        self.logger.info(
            f"Selector operation {self.id} selected {len(self.thoughts)}"
        )




