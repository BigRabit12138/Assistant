from datashaper import DEFAULT_INPUT_NAME

from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_base_text_units"


def build_steps(
        config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    chunk_column_name = config.get("chunk_column", "chunk")
    chunk_by_columns = config.get("chunk_by", []) or []
    n_tokens_column_name = config.get("n_tokens_column", "n_tokens")

    return [
        # 按id升序排序
        # text: str 输入文本
        # id: str 文本的Hash ID
        # title: str 文本的文件名
        # 输入：text, id, title
        # 输出：text, id, title
        {
            "verb": "orderby",
            "args": {
                "orders": [
                    {"column": "id", "direction": "asc"},
                ]
            },
            "input": {"source": DEFAULT_INPUT_NAME},
        },
        # 将id和text打包，变成text_with_ids新增列，
        # 元素内容(id, 文本)
        # 输入：text, id, title
        # 输出：text, id, title, text_with_ids
        {
            "verb": "zip",
            "args": {
                "columns": ["id", "text"],
                "to": "text_with_ids",
            },
        },
        # 使用id列分组，id将作为索引，将一个组内的text_with_ids合并到一个列表，并忽略其他列，最后重命名为texts，
        # 内容[(id, 文本)]
        # 输入：id, title, text, text_with_ids
        # 输出：id, texts
        {
            "verb": "aggregate_override",
            "args": {
                "groupby": [*chunk_by_columns] if len(chunk_by_columns) > 0 else None,
                "aggregations": [
                    {
                        "column": "text_with_ids",
                        "operation": "array_agg",
                        "to": "texts",
                    }
                ],
            },
        },
        # 对texts文本分块，分块后的文本序列存储到chunks列，
        # chunks列内容为[([id], 文本块, token数量)]
        # 输入：id, texts
        # 输出：id, texts, chunks
        {
            "verb": "chunk",
            "args": {
                "column": "texts",
                "to": "chunks",
                **config.get("text_chunk", {}),
            },
        },
        # 选择指定的列，
        # chunks列内容为[([id], 文本块, token数量)]
        # 输入：id, texts, chunks
        # 输出：id, chunks
        {
            "verb": "select",
            "args": {
                "columns": [*chunk_by_columns, "chunks"],
            },
        },
        # 将chunks列的列表解开，每行一个文本块，
        # 内容([id], 文本块, token数量)
        # 输入：id, chunks
        # 输出：id, chunks
        {
            "verb": "unroll",
            "args": {
                "column": "chunks",
            },
        },
        # 将chunks重命名为chunk,
        # chunk内容([id], 文本块, token数量)
        # 输入：id, chunks
        # 输出：id, chunk
        {
            "verb": "rename",
            "args": {
                "columns": {
                    "chunks": chunk_column_name,
                }
            },
        },
        # 为每一行使用chunk的内容生成hash ID，存入chunk_id列
        # chunk内容([id], 文本块, token数量)
        # 输入：id, chunk
        # 输出：id, chunk, chunk_id
        {
            "verb": "genid",
            "args": {
                "to": "chunk_id",
                "method": "md5_hash",
                "hash": [chunk_column_name],
            },
        },
        # 将chunk列解包成document_ids, chunk, n_tokens
        # document_ids列内容[id]
        # 输入：id, chunk, chunk_id
        # 输出：id, chunk, chunk_id, document_ids, n_tokens
        {
            "verb": "unzip",
            "args": {
                "column": chunk_column_name,
                "to": ["document_ids", chunk_column_name, n_tokens_column_name],
            },
        },
        # 将chunk_id列复制到id
        # document_ids列内容[id]
        # 输入：id, chunk, chunk_id, document_ids, n_tokens
        # 输出：id, chunk, chunk_id, document_ids, n_tokens
        {
            "verb": "copy",
            "args": {
                "column": "chunk_id",
                "to": "id",
            }
        },
        # 将chunk的值为空的过滤掉，并重置索引
        # id: str chunk文本的Hash ID
        # chunk: str 一块文本
        # chunk_id: str chunk文本的Hash ID
        # document_ids: list[str] [块内容来源文档的ID]
        # n_tokens: int 块的token数量
        # 输入：id, chunk, chunk_id, document_ids, n_tokens
        # 输出：id, chunk, chunk_id, document_ids, n_tokens
        {
            "verb": "filter",
            "args": {
                "column": chunk_column_name,
                "criteria": [
                    {
                        "type": "value",
                        "operator": "is not empty",
                    }
                ],
            },
        },
    ]
