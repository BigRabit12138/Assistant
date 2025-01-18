from pydantic import BaseModel, Field

import assistant.memory.graphrag_v1.config.defaults as defaults


class UmapConfig(BaseModel):
    enabled: bool = Field(
        description="A flag indicating whether to enable UMAP.",
        default=defaults.UMAP_ENABLED,
    )
