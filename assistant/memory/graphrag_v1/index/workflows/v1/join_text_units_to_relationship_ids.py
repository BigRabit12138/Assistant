from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "join_text_units_to_relationship_ids"


def build_steps(
        _config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    return [
        # 选择id, text_unit_ids列
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
        # 输出：id, text_unit_ids
        {
            "verb": "select",
            "args": {"columns": ["id", "text_unit_ids"]},
            "input": {"source": "workflow:create_final_relationships"},
        },
        # 展开text_unit_ids
        # 输入：id, text_unit_ids
        # 输出：id, text_unit_ids
        {
            "verb": "unroll",
            "args": {
                "column": "text_unit_ids",
            },
        },
        # 按text_unit_ids列分组，id列保留不同合并成列表，改名为relationship_ids，
        # text_unit_ids选第一个，改名为id
        # 输入：id, text_unit_ids
        # 输出：text_unit_ids, relationship_ids, id
        # relationship_ids: list[str] 一个文本块包含的边的ID
        {
            "verb": "aggregate_override",
            "args": {
                "groupby": ["text_unit_ids"],
                "aggregations": [
                    {
                        "column": "id",
                        "operation": "array_agg_distinct",
                        "to": "relationship_ids",
                    },
                    {
                        "column": "text_unit_ids",
                        "operation": "any",
                        "to": "id",
                    },
                ],
            },
        },
        # 选择id, relationship_ids列
        # 输入：text_unit_ids, relationship_ids, id
        # 输出：id, relationship_ids
        # id: str 文本块的ID
        # relationship_ids: list[str] 一个文本块包含的边的ID
        {
            "id": "text_unit_id_to_relationship_ids",
            "verb": "select",
            "args": {"columns": ["id", "relationship_ids"]},
        },
    ]
