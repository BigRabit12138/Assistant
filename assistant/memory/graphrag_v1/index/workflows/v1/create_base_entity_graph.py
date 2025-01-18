from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_base_entity_graph"


def build_steps(
        config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    clustering_config = config.get(
        "cluster_graph",
        {"strategy": {"type": "leiden"}},
    )
    embed_graph_config = config.get(
        "embed_graph",
        {
            "strategy": {
                "type": "node2vec",
                "num_walks": config.get("embed_num_walks", 10),
                "walk_length": config.get("embed_walk_length", 40),
                "window_size": config.get("embed_window_size", 2),
                "iterations": config.get("embed_iterations", 3),
                "random_seed": config.get("embed_random_seed", 86),
            }
        },
    )

    graphml_snapshot_enabled = config.get("graphml_snapshot", False) or False
    embed_graph_enabled = config.get("embed_graph_enabled", False) or False

    return [
        {
            # 对图的节点进行分层聚类
            # entity_graph: str 所有文本的graphml
            # 节点的属性(key, type, description, source_id)
            # 边的属性(source, target, weight, description, source_id)
            # 输入：entity_graph
            # 输出：entity_graph, level, clustered_graph
            # level: int 内容为层次
            # clustered_graph: str 内容为更新了本层节点的聚类属性的graphml图
            # 节点的属性(key, type, description, source_id, cluster(本层有), level(本层有), degree, human_readable_id, id)
            # 边的属性(source, target, weight, description, source_id, id, human_readable_id, level)
            "verb": "cluster_graph",
            "args": {
                **clustering_config,
                "column": "entity_graph",
                "to": "clustered_graph",
                "level_to": "level",
            },
            "input": ({
                "source": "workflow:create_summarized_entities"
            }),
        },
        # 默认关闭
        {
            "verb": "snapshot_rows",
            "enabled": graphml_snapshot_enabled,
            "args": {
                "base_name": "clustered_graph",
                "column": "clustered_graph",
                "formats": [{
                    "format": "text",
                    "extension": "graphml"
                }],
            },
        },
        # 默认关闭
        {
            "verb": "embed_graph",
            "enabled": embed_graph_enabled,
            "args": {
                "column": "clustered_graph",
                "to": "embeddings",
                **embed_graph_config,
            },
        },
        # 默认关闭
        {
            "verb": "snapshot_rows",
            "enabled": graphml_snapshot_enabled,
            "args": {
                "base_name": "embedded_graph",
                "column": "entity_graph",
                "formats": [{
                    "format": "text",
                    "extension": "graphml"
                }],
            },
        },
        # 选择聚类信息的列
        # 输入：entity_graph, level, clustered_graph
        # 输出：level, clustered_graph
        # level: int 内容为层次
        # clustered_graph: str 内容为更新了本层节点的聚类属性的graphml图
        # 节点的属性(key, type, description, source_id, cluster(本层有), level(本层有), degree, human_readable_id, id)
        # 边的属性(source, target, weight, description, source_id, id, human_readable_id, level)
        {
            "verb": "select",
            "args": {
                "columns": (
                    ["level", "clustered_graph", "embeddings"]
                    if embed_graph_enabled
                    else ["level", "clustered_graph"]
                ),
            },
        },
    ]
