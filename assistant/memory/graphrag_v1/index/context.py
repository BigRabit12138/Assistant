from dataclasses import field
from dataclasses import dataclass as dc_dataclass

from assistant.memory.graphrag_v1.index.cache import PipelineCache
from assistant.memory.graphrag_v1.index.storage.typing import PipelineStorage


@dc_dataclass
class PipelineRunStats:
    total_runtime: float = field(default=0)
    num_documents: int = field(default=0)
    input_load_time: float = field(default=0)
    workflows: dict[str, dict[str, float]] = field(default_factory=dict)


@dc_dataclass
class PipelineRunContext:
    stats: PipelineRunStats
    storage: PipelineStorage
    cache: PipelineCache


VerbRunContext = PipelineRunContext
