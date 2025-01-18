from datashaper import AsyncType
from pydantic import BaseModel, Field

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.config.models.llm_parameters import LLMParameters
from assistant.memory.graphrag_v1.config.models.parallelization_parameters import ParallelizationParameters


class LLMConfig(BaseModel):
    llm: LLMParameters = Field(
        description="The LLM configuration to use.",
        default=LLMParameters()
    )
    parallelization: ParallelizationParameters = Field(
        description="The parallelization configuration to use.",
        default=ParallelizationParameters(),
    )
    async_mode: AsyncType = Field(
        description="The async mode to use.",
        default=defaults.ASYNC_MODE,
    )
