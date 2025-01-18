from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_final_text_units"


def build_steps(
        config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    base_text_embed = config.get("text_embed", {})
    text_unit_text_embed_config = config.get("text_unit_text_embed", base_text_embed)
    # 默认关闭，False
    covariates_enabled = config.get("covariates_enabled", False)
    # 默认忽略，True
    skip_text_unit_embedding = config.get("skip_text_unit_embedding", False)
    is_using_vector_store = (
        text_unit_text_embed_config.get("strategy", {}).get("vector_store", None)
        is not None
    )

    return [
        # 选择id, chunk, document_ids, n_tokens
        # id: str chunk文本的Hash ID
        # chunk: str 一块文本
        # chunk_id: str chunk文本的Hash ID
        # document_ids: list[str] [块内容来源文档的ID]
        # n_tokens: int 块的token数量
        # 输入：id, chunk, chunk_id, document_ids, n_tokens
        # 输出：id, chunk, document_ids, n_tokens
        {
            "verb": "select",
            "args": {"columns": ["id", "chunk", "document_ids", "n_tokens"]},
            "input": {"source": "workflow:create_base_text_units"},
        },
        # 将chunk改名为text
        # 输入：id, chunk, document_ids, n_tokens
        # 输出：id, text, document_ids, n_tokens
        {
            "id": "pre_entity_join",
            "verb": "rename",
            "args": {
                "columns": {
                    "chunk": "text",
                },
            },
        },
        # 在id上以左连接的方式连接两个表
        # text_unit_ids: str 文本块id
        # entity_ids: list[str] 文本块所包含的实体的ID列表
        # id: 文本块id
        # 输入：id, text, document_ids, n_tokens
        # 输入：text_unit_ids, entity_ids, id
        # 输出：id, text, document_ids, n_tokens, text_unit_ids, entity_ids
        {
            "id": "pre_relationship_join",
            "verb": "join",
            "args": {
                "on": ["id", "id"],
                "strategy": "left outer",
            },
            "input": {
                "source": "pre_entity_join",
                "others": ["workflow:join_text_units_to_entity_ids"],
            },
        },
        # 在id上以左连接的方式连接两个表
        # id: str 文本块的ID
        # relationship_ids: list[str] 一个文本块包含的边的ID
        # 输入：id, text, document_ids, n_tokens, text_unit_ids, entity_ids
        # 输入：id, relationship_ids
        # 输出：id, text, document_ids, n_tokens, text_unit_ids, entity_ids, relationship_ids
        {
            "id": "pre_covariate_join",
            "verb": "join",
            "args": {
                "on": ["id", "id"],
                "strategy": "left outer",
            },
            "input": {
                "source": "pre_relationship_join",
                "others": ["workflow:join_text_units_to_relationship_ids"],
            },
        },
        # 默认关闭
        {
            "enabled": covariates_enabled,
            "verb": "join",
            "args": {
                "on": ["id", "id"],
                "strategy": "left outer",
            },
            "input": {
                "source": "pre_covariate_join",
                "others": ["workflow:join_text_units_to_covariate_ids"],
            },
        },
        # 按id分组，text选第一个，n_tokens，document_ids，entity_ids，relationship_ids同样
        # 输入：id, text, document_ids, n_tokens, text_unit_ids, entity_ids, relationship_ids
        # 输出：id, text, n_tokens，document_ids，entity_ids，relationship_ids
        {
            "verb": "aggregate_override",
            "args": {
                "groupby": ["id"],
                "aggregations": [
                    {
                        "column": "text",
                        "operation": "any",
                        "to": "text",
                    },
                    {
                        "column": "n_tokens",
                        "operation": "any",
                        "to": "n_tokens",
                    },
                    {
                        "column": "document_ids",
                        "operation": "any",
                        "to": "document_ids",
                    },
                    {
                        "column": "entity_ids",
                        "operation": "any",
                        "to": "entity_ids",
                    },
                    {
                        "column": "relationship_ids",
                        "operation": "any",
                        "to": "relationship_ids",
                    },
                    *(
                        []
                        if not covariates_enabled
                        else [
                            {
                                "column": "covariate_ids",
                                "operation": "any",
                                "to": "covariate_dis",
                            }
                        ]
                    ),
                ],
            },
        },
        # 默认关闭
        {
            "id": "embedded_text_units",
            "verb": "text_embed",
            "enabled": not skip_text_unit_embedding,
            "args": {
                "column": config.get("column", "text"),
                "to": config.get("to", "text_embedding"),
                **text_unit_text_embed_config,
            },
        },
        # 选id, text, n_tokens, document_ids, entity_ids, relationship_ids列
        # id: str 文本块ID
        # text: str 文本块内容
        # n_tokens: int 文本块token数量
        # document_ids: list[str] [块内容来源文档的ID]
        # entity_ids: list[str] 文本块所包含的实体的ID列表
        # relationship_ids: list[str] 一个文本块包含的边的ID
        # 输入：id, text, n_tokens，document_ids，entity_ids，relationship_ids
        # 输出：id, text, n_tokens，document_ids，entity_ids，relationship_ids
        {
            "verb": "select",
            "args": {
                "columns": [
                    "id",
                    "text",
                    *(
                        []
                        if (skip_text_unit_embedding or is_using_vector_store)
                        else ["text_embedding"]
                    ),
                    "n_tokens",
                    "document_ids",
                    "entity_ids",
                    "relationship_ids",
                    *([] if not covariates_enabled else ["covariate_ids"]),
                ],
            },
        },
    ]
