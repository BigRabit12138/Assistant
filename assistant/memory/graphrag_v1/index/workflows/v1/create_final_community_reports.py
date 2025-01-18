from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_final_community_reports"


def build_steps(
        config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    # 默认关闭，False
    covariates_enabled = config.get("covariates_enabled", False)
    create_community_reports_config = config.get("create_community_reports", {})
    base_text_embed = config.get("text_embed", {})
    community_report_full_content_embed_config = config.get(
        "community_report_full_content_embed", base_text_embed
    )
    community_report_summary_embed_config = config.get(
        "community_report_summary_embed", base_text_embed
    )
    community_report_title_embed_config = config.get(
        "community_report_title_embed", base_text_embed
    )
    # 默认忽略，True
    skip_title_embedding = config.get("skip_title_embedding", False)
    # 默认忽略，True
    skip_summary_embedding = config.get("skip_summary_embedding", False)
    # 默认忽略，True
    skip_full_content_embedding = config.get("skip_full_content_embedding", False)

    return [
        # 提取出节点的human_readable_id, title, description, degree属性
        # 赋值给node_details列
        # level: int 节点所属的层
        # title: str 节点的名称
        # #type: str 节点的类型
        # description: str 节点描述
        # source_id: str 节点源文本chunk_id
        # degree: int 节点的度
        # human_readable_id: str 节点可读ID
        # id: str 节点ID
        # size: int 节点的度
        # graph_embedding: list[float] 节点的向量
        # community: int 节点的聚簇
        # top_level_node_id: str 节点的ID
        # x: float 坐标X轴
        # y: float 坐标Y轴
        # 输入：level, title, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, community, top_level_node_id, x, y
        # 输出：level, title, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, community, top_level_node_id, x, y, node_details
        # node_details: dict[str, str] {"human_readable_id": 可读ID, "title": 名称, "description": 描述, "degree": 度}
        {
            "id": "nodes",
            "verb": "prepare_community_reports_nodes",
            "input": {"source": "workflow:create_final_nodes"},
        },
        # 获取边的human_readable_id, source, target, description, rank属性字典，赋值给
        # edge_details列
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
        # 输入：source, target, weight, description, text_unit_ids, id, human_readable_id,
        # source_degree, target_degree, rank
        # 输出：source, target, weight, description, text_unit_ids, id, human_readable_id,
        # source_degree, target_degree, rank, edge_details
        # edge_details: dict[str, str] {human_readable_id, source, target, description, rank}
        {
            "id": "edges",
            "verb": "prepare_community_reports_edges",
            "input": {"source": "workflow:create_final_relationships"},
        },
        # 默认忽略
        {
            "id": "claims",
            "enabled": covariates_enabled,
            "verb": "prepare_community_reports_claims",
            "input": {
                "source": "workflow:create_final_covariates",
            }
            # TODO：这个有用不？
            if covariates_enabled
            else {},
        },
        # 找到图中节点分层聚簇中，层间的所有community的包含关系
        # 输入：level, title, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, community, top_level_node_id, x, y, node_details
        # 输出：community, level, sub_community, sub_community_size
        {
            "id": "community_hierarchy",
            "verb": "restore_community_hierarchy",
            "input": {"source": "nodes"},
        },
        # 获取所有层的所有聚簇的对象的详细文本信息
        # 输入：level, title, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, community, top_level_node_id, x, y, node_details
        # 输入：source, target, weight, description, text_unit_ids, id, human_readable_id,
        # source_degree, target_degree, rank, edge_details
        # 输出：community, all_context, context_string, context_size,
        # context_exceed_limit, level
        {
            "id": "local_contexts",
            "verb": "prepare_community_reports",
            "input": {
                "source": "nodes",
                "nodes": "nodes",
                "edges": "edges",
                **({"claims": "claims"} if covariates_enabled else {}),
            },
        },
        # 获取所有聚簇的摘要信息
        # 输入：community, all_context, context_string, context_size,
        # context_exceed_limit, level
        # 输入：community, level, sub_community, sub_community_size
        # 输入：level, title, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, community, top_level_node_id, x, y, node_details
        # 输出：community, full_content, level, rank, title, rank_explanation, summary,
        # findings, full_content_json
        {
            "verb": "create_community_reports",
            "args": {
                **create_community_reports_config,
            },
            "input": {
                "source": "local_contexts",
                "community_hierarchy": "community_hierarchy",
                "nodes": "nodes",
            },
        },
        # 在id列生成uuid
        # community: str 聚簇ID
        # full_content: str 所有的聚簇检测结果
        # level: int 层次
        # rank: int 排序
        # title: str 标题
        # rank_explanation: str 排序理由
        # summary: str 摘要
        # findings: list[dict] 发现的内在关系
        # full_content_json: dict[str, str] 所有的聚簇检测结果json
        # id: str 检测结果UUID
        # 输入：community, full_content, level, rank, title, rank_explanation, summary,
        # findings, full_content_json
        # 输出：community, full_content, level, rank, title, rank_explanation, summary,
        # findings, full_content_json, id
        {
            "verb": "window",
            "args": {
                "to": "id",
                "operation": "uuid",
                "column": "community"
            },
        },
        # 默认忽略
        {
            "verb": "text_embed",
            "enabled": not skip_full_content_embedding,
            "args": {
                "embedding_name": "community_report_full_content",
                "column": "full_content",
                "to": "full_content_embedding",
                **community_report_full_content_embed_config,
            },
        },
        # 默认忽略
        {
            "verb": "text_embed",
            "enabled": not skip_summary_embedding,
            "args": {
                "embedding_name": "community_report_summary",
                "column": "summary",
                "to": "summary_embedding",
                **community_report_summary_embed_config,
            },
        },
        # 默认忽略
        {
            "verb": "text_embed",
            "enabled": not skip_title_embedding,
            "args": {
                "embedding_name": "community_report_title",
                "column": "title",
                "to": "title_embedding",
                **community_report_title_embed_config,
            },
        },
    ]
