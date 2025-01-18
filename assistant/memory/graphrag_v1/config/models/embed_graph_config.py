from pydantic import BaseModel, Field

import assistant.memory.graphrag_v1.config.defaults as defaults


class EmbedGraphConfig(BaseModel):
    enabled: bool = Field(
        description="A flag indicating whether to enable node2vec.",
        default=defaults.NODE2VEC_ENABLED,
    )
    num_walks: int = Field(
        description="The node2vec number of walks.",
        default=defaults.NODE2VEC_NUM_WALKS,
    )
    walk_length: int = Field(
        description="The node2vec walk length.",
        default=defaults.NODE2VEC_WALK_LENGTH,
    )
    window_size: int = Field(
        description="The node2vec window size.",
        default=defaults.NODE2VEC_WINDOW_SIZE,
    )
    iterations: int = Field(
        description="The node2vec iterations.",
        default=defaults.NODE2VEC_ITERATIONS,
    )
    random_seed: int = Field(
        description="The node2vec random seed.",
        default=defaults.NODE2VEC_RANDOM_SEED,
    )
    strategy: dict | None = Field(
        description="The graph embedding strategy override.",
        default=None,
    )

    def resolved_strategy(self) -> dict:
        from assistant.memory.graphrag_v1.index.verbs.graph.embed import EmbedGraphStrategyType

        return self.strategy or {
            "type": EmbedGraphStrategyType.node2vec,
            "num_walks": self.num_walks,
            "walk_length": self.walk_length,
            "window_size": self.window_size,
            "iterations": self.iterations,
            "random_seed": self.iterations,
        }
