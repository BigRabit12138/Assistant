from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_final_entities"


def build_steps(
        config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    base_text_embed = config.get("text_embed", {})
    entity_name_embed_config = config.get("entity_name_embed", base_text_embed)
    entity_name_description_embed_config = config.get(
        "entity_name_description_embed", base_text_embed
    )
    skip_name_embedding = config.get("skip_name_embedding", False)
    skip_description_embedding = config.get("skip_description_embedding", False)
    is_using_vector_store = (
        entity_name_embed_config.get("strategy", {}).get("vector_store", None)
        is not None
    )

    return [
        # 将图的节点数据解包，保留level列，每一行是一个节点
        # level: int 内容为层次
        # clustered_graph: str 内容为更新了本层节点的聚类属性的graphml图
        # 节点的属性(key, type, description, source_id, cluster(本层有), level(本层有), degree, human_readable_id, id)
        # 边的属性(source, target, weight, description, source_id, id, human_readable_id, level)
        # 输入：level, clustered_graph
        # 输出：level, label, type, description, source_id, degree, human_readable_id, id, graph_embedding,
        # cluster
        {
            "verb": "unpack_graph",
            "args": {
                "column": "clustered_graph",
                "type": "nodes",
            },
            "input": {"source": "workflow:create_base_entity_graph"},
        },
        # 修改label为title
        # 输入：level, label, type, description, source_id, degree, human_readable_id, id, graph_embedding,
        # cluster
        # 输出：level, title, type, description, source_id, degree, human_readable_id, id, graph_embedding,
        # cluster
        {
            "verb": "rename",
            "args": {
                "columns": {"label": "title"}
            },
        },
        # 保留指定的列
        # 输入：level, title, type, description, source_id, degree, human_readable_id, id, graph_embedding,
        # cluster
        # 输出：id, title, type, description, human_readable_id, graph_embedding, source_id
        {
            "verb": "select",
            "args": {
                "columns": [
                    "id",
                    "title",
                    "type",
                    "description",
                    "human_readable_id",
                    "graph_embedding",
                    "source_id",
                ],
            },
        },
        # 对id列去重
        # 输入：id, title, type, description, human_readable_id, graph_embedding, source_id
        # 输出：id, title, type, description, human_readable_id, graph_embedding, source_id
        {
            "verb": "dedupe",
            "args": {"columns": ["id"]},
        },
        # 将title列改为name（就不能一次改完，闹呢？）
        # 输入：id, title, type, description, human_readable_id, graph_embedding, source_id
        # 输出：id, name, type, description, human_readable_id, graph_embedding, source_id
        {
            "verb": "rename",
            "args": {
                "columns": {"title": "name"}
            }
        },
        # 对name列过滤，去掉空的
        # 输入：id, name, type, description, human_readable_id, graph_embedding, source_id
        # 输出：id, name, type, description, human_readable_id, graph_embedding, source_id
        {
            "verb": "filter",
            "args": {
                "column": "name",
                "criteria": [
                    {
                        "type": "value",
                        "operator": "is not empty",
                    }
                ],
            },
        },
        # 将source_id的内容分割成列表赋值给text_units_ids列
        # 输入：id, name, type, description, human_readable_id, graph_embedding, source_id
        # 输出：id, name, type, description, human_readable_id, graph_embedding, source_id, text_unit_ids
        # text_unit_ids: list[str] [节点源chunk_id]
        {
            "verb": "text_split",
            "args": {
                "separator": ",",
                "column": "source_id",
                "to": "text_unit_ids"
            },
        },
        # 去掉source_id列
        # 输入：id, name, type, description, human_readable_id, graph_embedding, source_id, text_unit_ids
        # 输出：id, name, type, description, human_readable_id, graph_embedding, text_unit_ids
        {
            "verb": "drop",
            "args": {"columns": ["source_id"]}
        },
        # 默认忽略
        {
            "verb": "text_embed",
            "enabled": not skip_name_embedding,
            "args": {
                "embedding_name": "entity_name",
                "column": "name",
                "to": "name_embedding",
                **entity_name_embed_config,
            },
        },
        # 将name和description列的内容使用:拼接，赋值给name_description
        # 输入：id, name, type, description, human_readable_id, graph_embedding, text_unit_ids
        # 输出：id, name, type, description, human_readable_id, graph_embedding, text_unit_ids, name_description
        {
            "verb": "merge",
            "enabled": not skip_description_embedding,
            "args": {
                "strategy": "concat",
                "columns": ["name", "description"],
                "to": "name_description",
                "delimiter": ":",
                "preserveSource": True,
            },
        },
        # 对name_description作embedding
        # 输入：id, name, type, description, human_readable_id, graph_embedding, text_unit_ids, name_description
        # 输出：id, name, type, description, human_readable_id, graph_embedding, text_unit_ids,
        # name_description, description_embedding
        # description_embedding: list[float] 向量
        {
            "verb": "text_embed",
            "enabled": not skip_description_embedding,
            "args": {
                "embedding_name": "entity_name_description",
                "column": "name_description",
                "to": "description_embedding",
                **entity_name_description_embed_config,
            },
        },
        # 去掉name_description列
        # 输入：id, name, type, description, human_readable_id, graph_embedding, text_unit_ids,
        # name_description, description_embedding
        # 输出：id, name, type, description, human_readable_id, graph_embedding, text_unit_ids, description_embedding
        {
            "verb": "drop",
            "enabled": not skip_description_embedding,
            "args": {
                "columns": ["name_description"],
            },
        },
        # 过滤description_embedding列中空的行
        # id: str 节点ID
        # name str 节点名称
        # #type: str 节点类型
        # description: str 节点描述
        # human_readable_id: int 节点可读ID
        # graph_embedding: list[float] 向量
        # text_unit_ids: str list[str] [节点源chunk_id]
        # description_embedding: list[float] 向量
        # 输入：id, type, human_readable_id, graph_embedding, text_unit_ids, description_embedding
        # 输出：id, type, human_readable_id, graph_embedding, text_unit_ids, description_embedding
        {
            "verb": "filter",
            "enabled": not skip_description_embedding and not is_using_vector_store,
            "args": {
                "column": "description_embedding",
                "criteria": [
                    {
                        "type": "value",
                        "operator": "is not empty",
                    }
                ],
            },
        },
    ]
