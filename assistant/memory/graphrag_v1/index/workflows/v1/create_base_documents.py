from datashaper import DEFAULT_INPUT_NAME

from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_base_documents"


def build_steps(
        config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    document_attribute_columns = config.get("document_attribute_columns", [])
    return [
        # 将document_ids列表展开，每行一个document_id
        # id: str 文本块ID
        # text: str 文本块内容
        # n_tokens: int 文本块token数量
        # document_ids: list[str] [块内容来源文档的ID]
        # entity_ids: list[str] 文本块所包含的实体的ID列表
        # relationship_ids: list[str] 一个文本块包含的边的ID
        # 输入：id, text, n_tokens, document_ids, entity_ids, relationship_ids
        # 输出：id, text, n_tokens, document_ids, entity_ids, relationship_ids
        {
            "verb": "unroll",
            "args": {"column": "document_ids"},
            "input": {"source": "workflow:create_final_text_units"},
        },
        # 选择id, document_ids, text
        # 输入：id, text, n_tokens, document_ids, entity_ids, relationship_ids
        # 输出：id, document_ids, text
        {
            "verb": "select",
            "args": {
                "columns": ["id", "document_ids", "text"]
            },
        },
        # 将document_ids改名为chunk_doc_id，id为chunk_id，text改为chunk_text
        # 输入：id, document_ids, text
        # 输出：chunk_id, chunk_doc_id, chunk_text
        {
            "id": "rename_chunk_doc_id",
            "verb": "rename",
            "args": {
                "columns": {
                    "document_ids": "chunk_doc_id",
                    "id": "chunk_id",
                    "text": "chunk_text",
                }
            },
        },
        # 将两个表在chunk_doc_id和id上使用内连接合并
        # 输入：chunk_id, chunk_doc_id, chunk_text
        # 输入：text, id, title
        # 输出：chunk_id, chunk_doc_id, chunk_text, text, id, title
        {
            "verb": "join",
            "args": {
                "on": ["chunk_doc_id", "id"]
            },
            "input": {
                "source": "rename_chunk_doc_id",
                "others": [DEFAULT_INPUT_NAME]
            },
        },
        # 按id分组，将chunk_id聚合为列表并改名为text_units
        # 输入：chunk_id, chunk_doc_id, chunk_text, text, id, title
        # 输出：id, text_units
        # text_units: list[str] [文本块id]
        {
            "id": "docs_with_text_units",
            "verb": "aggregate_override",
            "args": {
                "groupby": ["id"],
                "aggregations": [
                    {
                        "column": "chunk_id",
                        "operation": "array_agg",
                        "to": "text_units",
                    }
                ],
            },
        },
        # 将两个表在id上使用右连接合并
        # 输入：id, text_units
        # 输入：text, id, title
        # 输出：id, text_units, text, title
        {
            "verb": "join",
            "args": {
                "on": ["id", "id"],
                "strategy": "right outer",
            },
            "input": {
                "source": "docs_with_text_units",
                "others": [DEFAULT_INPUT_NAME],
            },
        },
        # 将text改名为raw_content
        # 输入：id, text_units, text, title
        # 输出：id, text_units, raw_content, title
        {
            "verb": "rename",
            "args": {"columns": {"text": "raw_content"}},
        },
        *[
            # 默认忽略
            {
                "verb": "convert",
                "args": {
                    "column": column,
                    "to": column,
                    "type": "string",
                },
            }
            for column in document_attribute_columns
        ],
        # 默认忽略
        {
            "verb": "merge_override",
            "enabled": len(document_attribute_columns) > 0,
            "args": {
                "columns": document_attribute_columns,
                "strategy": "json",
                "to": "attributes",
            },
        },
        # 将id转为string
        # id: str 文档ID
        # text_units: list[str] [文档包含的文本块ID]
        # raw_context: str 文档文本内容
        # title: str 文档文件名
        # 输入：id, text_units, raw_content, title
        # 输出：id, text_units, raw_content, title
        {
            "verb": "convert",
            "args": {"column": "id", "to": "id", "type": "string"}
        },
    ]
