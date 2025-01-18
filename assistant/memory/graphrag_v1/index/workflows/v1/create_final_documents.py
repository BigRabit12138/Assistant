from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_final_documents"


def build_steps(
        config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    base_text_embed = config.get("text_embed", {})
    document_raw_content_embed_config = config.get(
        "document_raw_content_embed", base_text_embed
    )
    # 默认忽略，True
    skip_raw_content_embedding = config.get(
        "skip_raw_content_embedding", False
    )

    return [
        # 将text_units改名为text_unit_ids
        # id: str 文档ID
        # text_units: list[str] [文档包含的文本块ID]
        # raw_context: str 文档文本内容
        # title: str 文档文件名
        # 输入：id, text_units, raw_content, title
        # 输入：id, text_unit_ids, raw_content, title
        {
            "verb": "rename",
            "args": {"columns": {"text_units": "text_unit_ids"}},
            "input": {"source": "workflow:create_base_documents"},
        },
        # 默认忽略
        {
            "verb": "text_embed",
            "enabled": not skip_raw_content_embedding,
            "args": {
                "column": "raw_content",
                "to": "raw_content_embedding",
                **document_raw_content_embed_config,
            },
        },
    ]
