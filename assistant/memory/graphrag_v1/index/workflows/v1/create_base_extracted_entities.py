from datashaper import AsyncType

from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_base_extracted_entities"


def build_steps(
        config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    entity_extraction_config = config.get("entity_extract", {})
    graphml_snapshot_enabled = config.get("graphml_snapshot", False) or False
    raw_entity_snapshot_enabled = config.get("raw_entity_snapshot", False) or False

    return [
        # 对分块的文本抽取实体
        # id: str chunk文本的Hash ID
        # chunk: str 一块文本
        # chunk_id: str chunk文本的Hash ID
        # document_ids: list[str] [块内容来源文档的ID]
        # n_tokens: int 块的token数量
        # 输入：id, chunk, chunk_id, document_ids, n_tokens
        # 输出：id, chunk, chunk_id, document_ids, n_tokens, entities, entity_graph
        # entities: list[dict[str: str]]
        # 一个文本块中所有实体[一个实体{"name": 实体名称, "type": 实体类型, "description": 实体描述, "source_id": 文本块ID}]
        # entity_graph: str 一个文本块图的graphml
        # 节点的属性(key, type, description, source_id)
        # 边的属性(source, target, weight, description, source_id)
        {
            "verb": "entity_extract",
            "args": {
                **entity_extraction_config,
                "column": entity_extraction_config.get("text_column", "chunk"),
                "id_column": entity_extraction_config.get("id_column", "chunk_id"),
                "async_mode": entity_extraction_config.get(
                    "async_mode", AsyncType.AsyncIO
                ),
                "to": "entities",
                "graph_to": "entity_graph",
            },
            "input": {"source": "workflow:create_base_text_units"},
        },
        # 默认关闭
        {
            "verb": "snapshot",
            "enabled": raw_entity_snapshot_enabled,
            "args": {
                "name": "raw_extracted_entities",
                "formats": ["json"],
            },
        },
        # 将每个块的子图合并到一个大图
        # entity_graph: str 所有文本的graphml
        # 节点的属性(key, type, description, source_id)
        # 边的属性(source, target, weight, description, source_id)
        # 输入：id, chunk_id, document_ids, chunk, n_tokens, entities, entity_graph
        # 输出：entity_graph
        {
            "verb": "merge_graphs",
            "args": {
                "column": "entity_graph",
                "to": "entity_graph",
                **config.get(
                    "graph_merge_operations",
                    {
                        "nodes": {
                            "source_id": {
                                "operation": "concat",
                                "delimiter": ", ",
                                "distinct": True,
                            },
                            "description": ({
                                "operation": "concat",
                                "separator": "\n",
                                "distinct": False,
                            }),
                        },
                        "edges": {
                            "source_id": {
                                "operation": "concat",
                                "delimiter": ", ",
                                "distinct": True,
                            },
                            "description": ({
                                "operation": "concat",
                                "separator": "\n",
                                "distinct": False,
                            }),
                            "weight": "sum",
                        },
                    },
                ),
            },
        },
        # 默认关闭
        {
            "verb": "snapshot_rows",
            "enabled": graphml_snapshot_enabled,
            "args": {
                "base_name": "merged_graph",
                "column": "entity_graph",
                "formats": [{
                    "format": "text",
                    "extension": "graphml"
                }],
            },
        },
    ]
