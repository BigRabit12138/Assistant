from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "join_text_units_to_entity_ids"


def build_steps(
        _config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    return [
        # 选择id, text_unit_ids列
        # id: str 节点ID
        # #type: str 节点类型
        # human_readable_id: int 节点可读ID
        # graph_embedding: list[float] 向量
        # text_unit_ids: str list[str] [节点源chunk_id]
        # description_embedding: list[float] 向量
        # 输入：id, type, human_readable_id, graph_embedding, text_unit_ids, description_embedding
        # 输出：id, text_unit_ids
        {
            "verb": "select",
            "args": {"columns": ["id", "text_unit_ids"]},
            "input": {"source": "workflow:create_final_entities"},
        },
        # 将text_unit_ids的列表解开
        # 输入：id, text_unit_ids
        # 输出：id, text_unit_ids
        {
            "verb": "unroll",
            "args": {
                "column": "text_unit_ids",
            },
        },
        # 按text_unit_ids分组，将id列和text_unit_ids聚合，id列保留不同组成列表
        # 改名为entity_ids，text_unit_ids选第一个，改名为id
        # 输入：id, text_unit_ids
        # 输出：text_unit_ids, entity_ids, id
        # text_unit_ids: str 文本块id
        # entity_ids: list[str] 文本块所包含的实体的ID列表
        # id: 文本块id
        {
            "verb": "aggregate_override",
            "args": {
                "groupby": ["text_unit_ids"],
                "aggregations": [
                    {
                        "column": "id",
                        "operation": "array_agg_distinct",
                        "to": "entity_ids",
                    },
                    {
                        "column": "text_unit_ids",
                        "operation": "any",
                        "to": "id"
                    },
                ],
            },
        },
    ]
