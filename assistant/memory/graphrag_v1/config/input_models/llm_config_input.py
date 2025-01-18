from datashaper import AsyncType
from typing_extensions import NotRequired, TypedDict

from assistant.memory.graphrag_v1.config.input_models.llm_parameters_input import LLMParametersInput
from assistant.memory.graphrag_v1.config.input_models.parallelization_parameters_input import \
    ParallelizationParametersInput


class LLMConfigInput(TypedDict):
    llm: NotRequired[LLMParametersInput | None]
    parallelization: NotRequired[ParallelizationParametersInput | None]
    async_mode: NotRequired[AsyncType | str | None]
