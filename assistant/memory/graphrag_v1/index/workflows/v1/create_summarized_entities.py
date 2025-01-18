from datashaper import AsyncType

from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_summarized_entities"


def build_steps(
        config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    summarize_descriptions_config = config.get("summarize_descriptions", {})
    graphml_snapshot_enabled = config.get("graphml_snapshot", False) or False

    return [
        # 摘要图对象的描述
        # entity_graph: str 所有文本的graphml
        # 节点的属性(key, type, description, source_id)
        # 边的属性(source, target, weight, description, source_id)
        # 输入：entity_graph
        # 输出：entity_graph
        {
            "verb": "summarize_descriptions",
            "args": {
                **summarize_descriptions_config,
                "column": "entity_graph",
                "to": "entity_graph",
                "async_mode": summarize_descriptions_config.get(
                    "async_mode", AsyncType.AsyncIO
                ),
            },
            "input": {"source": "workflow:create_base_extracted_entities"},
        },
        # 默认关闭
        {
            "verb": "snapshot_rows",
            "enabled": graphml_snapshot_enabled,
            "args": {
                "base_name": "summarized_graph",
                "column": "entity_graph",
                "formats": [{"format": "text", "extension": "graphml"}],
            },
        },
    ]
