from pydantic import BaseModel, Field

import assistant.memory.graphrag_v1.config.defaults as defaults


class SnapshotsConfig(BaseModel):
    graphml: bool = Field(
        description="A flag indicating whether to take snapshots of GraphML.",
        default=defaults.SNAPSHOTS_GRAPHML,
    )
    raw_entities: bool = Field(
        description="A flag indicating whether to take snapshots of raw entities.",
        default=defaults.SNAPSHOTS_RAW_ENTITIES,
    )
    top_level_nodes: bool = Field(
        description="A flag indicating whether to take snapshots of top-level nodes.",
        default=defaults.SNAPSHOTS_TOP_LEVEL_NODES,
    )
