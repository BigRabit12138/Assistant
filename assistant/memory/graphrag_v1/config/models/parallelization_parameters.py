from pydantic import BaseModel, Field

import assistant.memory.graphrag_v1.config.defaults as defaults


class ParallelizationParameters(BaseModel):
    stagger: float = Field(
        description="The stagger to use for the LLM service.",
        default=defaults.PARALLELIZATION_STAGGER,
    )
    num_threads: int = Field(
        description="The number of threads to use for the LLM service.",
        default=defaults.PARALLELIZATION_NUM_THREADS
    )
