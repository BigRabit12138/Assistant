from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_final_relationships"


def build_steps(
        config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    base_text_embed = config.get("text_embed", {})
    relationship_description_embed_config = config.get(
        "relationship_description_embed", base_text_embed
    )
    skip_description_embedding = config.get("skip_description_embedding", False)

    return [
        # 将图的边解包，每一行是一个边，保留level列
        # level: int 内容为层次
        # clustered_graph: str 内容为更新了本层节点的聚类属性的graphml图
        # 节点的属性(key, type, description, source_id, cluster(本层有), level(本层有), degree, human_readable_id, id)
        # 边的属性(source, target, weight, description, source_id, id, human_readable_id, level)
        # 输入：level, clustered_graph
        # 输出：level, source, target, weight, description, source_id, id, human_readable_id,
        {
            "verb": "unpack_graph",
            "args": {
                "column": "clustered_graph",
                "type": "edges",
            },
            "input": {"source": "workflow:create_base_entity_graph"},
        },
        # 将source_id改名为text_unit_ids
        # 输入：level, source, target, weight, description, source_id, id, human_readable_id,
        # 输出：level, source, target, weight, description, text_unit_ids, id, human_readable_id,
        {
            "verb": "rename",
            "args": {"columns": {"source_id": "text_unit_ids"}},
        },
        # 保留level为0的行
        # 输入：level, source, target, weight, description, text_unit_ids, id, human_readable_id,
        # 输出：level, source, target, weight, description, text_unit_ids, id, human_readable_id,
        {
            "verb": "filter",
            "args": {
                "column": "level",
                "criteria": [{"type": "value", "operator": "equals", "value": 0}],
            },
        },
        # 默认忽略
        {
            "verb": "text_embed",
            "enabled": not skip_description_embedding,
            "args": {
                "embedding_name": "relationship_description",
                "column": "description",
                "to": "description_embedding",
                **relationship_description_embed_config,
            },
        },
        # 舍弃level列
        # 输入：level, source, target, weight, description, text_unit_ids, id, human_readable_id,
        # 输出：source, target, weight, description, text_unit_ids, id, human_readable_id,
        {
            "id": "pruned_edges",
            "verb": "drop",
            "args": {"columns": ["level"]},
        },
        # 保留level为0的行
        # 输入：level, title, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, community, top_level_node_id, x, y
        # 输出：level, title, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, community, top_level_node_id, x, y
        {
            "id": "filtered_nodes",
            "verb": "filter",
            "args": {
                "column": "level",
                "criteria": [{"type": "value", "operator": "equals", "value": 0}],
            },
            "input": "workflow:create_final_nodes",
        },
        # 计算边的度
        # 输入：source, target, weight, description, text_unit_ids, id, human_readable_id,
        # 输入：level, title, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, community, top_level_node_id, x, y
        # 输出：source, target, weight, description, text_unit_ids, id, human_readable_id,
        # source_degree, target_degree, rank
        {
            "verb": "compute_edge_combined_degree",
            "args": {"to": "rank"},
            "input": {
                "source": "pruned_edges",
                "nodes": "filtered_nodes",
            },
        },
        # 将human_readable_id列转换为字符串
        # 输入：source, target, weight, description, text_unit_ids, id, human_readable_id,
        # source_degree, target_degree, rank
        # 输出：source, target, weight, description, text_unit_ids, id, human_readable_id,
        # source_degree, target_degree, rank
        {
            "verb": "convert",
            "args": {
                "column": "human_readable_id",
                "type": "string",
                "to": "human_readable_id",
            },
        },
        # 将text_unit_ids列转换为列表
        # 输入：source, target, weight, description, text_unit_ids, id, human_readable_id,
        # source_degree, target_degree, rank
        # 输出：source, target, weight, description, text_unit_ids, id, human_readable_id,
        # source_degree, target_degree, rank
        # source: str 边的源点
        # target: str 边的目的点
        # weight: float 边的权重
        # description: str 边的描述
        # text_unit_ids: lis[str] 边的点的源chunk_id
        # id: str 边的ID
        # human_readable_id: str 边的可读ID
        # source_degree: int 边的源点的度
        # target_degree: int 边的目的点的度
        # rank: int 边的度
        {
            "verb": "convert",
            "args": {
                "column": "text_unit_ids",
                "type": "array",
                "to": "text_unit_ids",
            },
        },
    ]
