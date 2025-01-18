from pydantic import BaseModel, Field

import assistant.memory.graphrag_v1.config.defaults as defaults


class ClusterGraphConfig(BaseModel):
    max_cluster_size: int = Field(
        description="The maximum cluster size to use.",
        default=defaults.MAX_CLUSTER_SIZE,
    )
    strategy: dict | None = Field(
        description="The cluster strategy to use.",
        default=None,
    )

    def resolved_strategy(self) -> dict:
        from assistant.memory.graphrag_v1.index.verbs.graph.clustering import GraphCommunityStrategyType

        return self.strategy or {
            "type": GraphCommunityStrategyType.leiden,
            "max_cluster_size": self.max_cluster_size,
        }
