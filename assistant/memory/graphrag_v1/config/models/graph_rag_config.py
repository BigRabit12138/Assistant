from pydantic import Field
from devtools import pformat

import assistant.memory.graphrag_v1.config.defaults as defaults

from assistant.memory.graphrag_v1.config.models.llm_config import LLMConfig
from assistant.memory.graphrag_v1.config.models.umap_config import UmapConfig
from assistant.memory.graphrag_v1.config.models.input_config import InputConfig
from assistant.memory.graphrag_v1.config.models.cache_config import CacheConfig
from assistant.memory.graphrag_v1.config.models.storage_config import StorageConfig
from assistant.memory.graphrag_v1.config.models.chunking_config import ChunkingConfig
from assistant.memory.graphrag_v1.config.models.reporting_config import ReportingConfig
from assistant.memory.graphrag_v1.config.models.snapshots_config import SnapshotsConfig
from assistant.memory.graphrag_v1.config.models.embed_graph_config import EmbedGraphConfig
from assistant.memory.graphrag_v1.config.models.local_search_config import LocalSearchConfig
from assistant.memory.graphrag_v1.config.models.cluster_graph_config import ClusterGraphConfig
from assistant.memory.graphrag_v1.config.models.global_search_config import GlobalSearchConfig
from assistant.memory.graphrag_v1.config.models.text_embedding_config import TextEmbeddingConfig
from assistant.memory.graphrag_v1.config.models.claim_extraction_config import ClaimExtractionConfig
from assistant.memory.graphrag_v1.config.models.community_reports_config import CommunityReportsConfig
from assistant.memory.graphrag_v1.config.models.entity_extraction_config import EntityExtractionConfig
from assistant.memory.graphrag_v1.config.models.summarize_descriptions_config import SummarizeDescriptionConfig


class GraphRagConfig(LLMConfig):
    def __repr__(self) -> str:
        return pformat(self, highlight=False)

    def __str__(self):
        # TODO: 啥子情况, 阿， 我忘记我为啥子写这个了，坏了，坏了
        return self.model_dump_json(indent=4)

    root_dir: str = Field(
        description="The root directory for the configuration.",
        default=None,
    )
    reporting: ReportingConfig = Field(
        description="The reporting configuration.",
        default=ReportingConfig(),
    )
    storage: StorageConfig = Field(
        description="The storage configuration.",
        default=StorageConfig(),
    )
    cache: CacheConfig = Field(
        description="The cache configuration.",
        default=CacheConfig(),
    )
    input: InputConfig = Field(
        description="The input configuration.",
        default=InputConfig(),
    )
    embed_graph: EmbedGraphConfig = Field(
        description="The input configuration.", default=InputConfig()
    )
    embeddings: TextEmbeddingConfig = Field(
        description="The embeddings LLM configuration to use.",
        default=TextEmbeddingConfig(),
    )
    chunks: ChunkingConfig = Field(
        description="The chunking configuration to use.",
        default=ChunkingConfig(),
    )
    snapshots: SnapshotsConfig = Field(
        description="The snapshots configuration to use.",
        default=SnapshotsConfig(),
    )
    entity_extraction: EntityExtractionConfig = Field(
        description="The entity extraction configuration to use.",
        default=EntityExtractionConfig(),
    )
    summarize_descriptions: SummarizeDescriptionConfig = Field(
        description="The description summarization configuration to use.",
        default=SummarizeDescriptionConfig(),
    )
    community_reports: CommunityReportsConfig = Field(
        description="The community reports configuration to use.",
        default=CommunityReportsConfig(),
    )
    claim_extraction: ClaimExtractionConfig = Field(
        description="The claim extraction configuration to use.",
        default=ClaimExtractionConfig(
            enabled=defaults.CLAIM_EXTRACTION_ENABLED,
        ),
    )
    cluster_graph: ClusterGraphConfig = Field(
        description="The cluster graph configuration to use.",
        default=ClusterGraphConfig(),
    )
    umap: UmapConfig = Field(
        description="The UMAP configuration to use.",
        default=UmapConfig(),
    )
    loca_search: LocalSearchConfig = Field(
        description="The local search configuration.",
        default=LocalSearchConfig(),
    )
    global_search: GlobalSearchConfig = Field(
        description="The global search configuration.",
        default=GlobalSearchConfig(),
    )
    encoding_model: str = Field(
        description="The encoding model to use.",
        default=defaults.ENCODING_MODEL,
    )
    skip_workflows: list[str] = Field(
        description="The workflows to ship, usually for testing reasons.",
        default=[],
    )
