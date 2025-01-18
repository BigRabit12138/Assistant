from typing_extensions import NotRequired

from assistant.memory.graphrag_v1.config.input_models.llm_config_input import LLMConfigInput
from assistant.memory.graphrag_v1.config.input_models.umap_config_input import UmapConfigInput
from assistant.memory.graphrag_v1.config.input_models.cache_config_input import CacheConfigInput
from assistant.memory.graphrag_v1.config.input_models.input_config_input import InputConfigInput
from assistant.memory.graphrag_v1.config.input_models.storage_config_input import StorageConfigInput
from assistant.memory.graphrag_v1.config.input_models.chunking_config_input import ChunkingConfigInput
from assistant.memory.graphrag_v1.config.input_models.reporting_config_input import ReportingConfigInput
from assistant.memory.graphrag_v1.config.input_models.snapshots_config_input import SnapshotsConfigInput
from assistant.memory.graphrag_v1.config.input_models.embed_graph_config_input import EmbedGraphConfigInput
from assistant.memory.graphrag_v1.config.input_models.local_search_config_input import LocalSearchConfigInput
from assistant.memory.graphrag_v1.config.input_models.cluster_graph_config_input import ClusterGraphConfigInput
from assistant.memory.graphrag_v1.config.input_models.global_search_config_input import GlobalSearchConfigInput
from assistant.memory.graphrag_v1.config.input_models.text_embedding_config_input import TextEmbeddingConfigInput
from assistant.memory.graphrag_v1.config.input_models.claim_extraction_config_input import ClaimExtractionConfigInput
from assistant.memory.graphrag_v1.config.input_models.community_reports_config_input import CommunityReportsConfigInput
from assistant.memory.graphrag_v1.config.input_models.entity_extraction_config_input import EntityExtractionConfigInput
from assistant.memory.graphrag_v1.config.input_models.summarize_descriptions_config_input import \
    SummarizeDescriptionConfigInput


class GraphRagConfigInput(LLMConfigInput):
    reporting: NotRequired[ReportingConfigInput | None]
    storage: NotRequired[StorageConfigInput | None]
    cache: NotRequired[CacheConfigInput | None]
    input: NotRequired[InputConfigInput | None]
    embed_graph: NotRequired[EmbedGraphConfigInput | None]
    embeddings: NotRequired[TextEmbeddingConfigInput | None]
    chunks: NotRequired[ChunkingConfigInput | None]
    snapshots: NotRequired[SnapshotsConfigInput | None]
    entity_extraction: NotRequired[EntityExtractionConfigInput | None]
    summarize_descriptions: NotRequired[SummarizeDescriptionConfigInput | None]
    community_reports: NotRequired[CommunityReportsConfigInput | None]
    claim_extraction: NotRequired[ClaimExtractionConfigInput | None]
    cluster_graph: NotRequired[ClusterGraphConfigInput | None]
    umap: NotRequired[UmapConfigInput | None]
    encoding_model: NotRequired[str | None]
    skip_workflows: NotRequired[list[str] | str | None]
    local_search: NotRequired[LocalSearchConfigInput | None]
    global_search: NotRequired[GlobalSearchConfigInput | None]
