from dataclasses import dataclass

from assistant.memory.graphrag_v1.model.identified import Identified


@dataclass
class Named(Identified):
    title: str
