from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_final_nodes"


def build_steps(
        config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    # 默认关闭
    snapshot_top_level_nodes = config.get("snapshot_top_level_nodes", False)
    # 默认关闭
    layout_graph_enabled = config.get("layout_graph_enabled", True)
    _compute_top_level_positions = [
        # 将带位置信息的图的节点解包，每一行是一个节点，保留level列
        # 输入：level, clustered_graph, node_positions, positioned_graph
        # 输出：level, label, type, description, source_id, degree, human_readable_id, id, x, y,
        # size, graph_embedding, cluster
        # TODO: 这难道不是多余？
        {
            "verb": "unpack_graph",
            "args": {"column": "positioned_graph", "type": "nodes"},
            "input": {"source": "laid_out_entity_graph"},
        },
        # 保留level为0的节点
        # 输入：level, label, type, description, source_id, degree, human_readable_id, id, x, y,
        # size, graph_embedding, cluster
        # 输出：level, label, type, description, source_id, degree, human_readable_id, id, x, y,
        # size, graph_embedding, cluster
        {
            "verb": "filter",
            "args": {
                "column": "level",
                "criteria": [
                    {
                        "type": "value",
                        "operator": "equals",
                        "value": config.get("level_for_node_positions", 0),
                    }
                ],
            },
        },
        # 选择id，x，y这三列
        # 输入：level, label, type, description, source_id, degree, human_readable_id, id, x, y,
        # size, graph_embedding, cluster
        # 输出: id, x, y
        {
            "verb": "select",
            "args": {"columns": ["id", "x", "y"]},
        },
        # 默认关闭
        {
            "verb": "snapshot",
            "enabled": snapshot_top_level_nodes,
            "args": {
                "name": "top_level_nodes",
                "formats": ["json"],
            },
        },
        # 将id列改名为top_level_node_id
        # 输入: id, x, y
        # 输出: top_level_node_id, x, y
        {
            "id": "_compute_top_level_node_positions",
            'verb': "rename",
            "args": {
                "columns": {
                    "id": "top_level_node_id",
                }
            },
        },
        # 将top_level_node_id转为string
        # 输入: top_level_node_id, x, y
        # 输出: top_level_node_id, x, y
        {
            "verb": "convert",
            "args": {
                "column": "top_level_node_id",
                "to": "top_level_node_id",
                "type": "string",
            },
        },
    ]
    layout_graph_config = config.get(
        "layout_graph",
        {
            "strategy": {
                "type": "umap" if layout_graph_enabled else "zero",
            },
        },
    )

    return [
        # 获取图的节点的空间位置布局
        # level: int 内容为层次
        # clustered_graph: str 内容为更新了本层节点的聚类属性的graphml图
        # 节点的属性(key, type, description, source_id, cluster(本层有), level(本层有), degree, human_readable_id, id)
        # 边的属性(source, target, weight, description, source_id, id, human_readable_id, level)
        # 输入：level, clustered_graph
        # 输出：level, clustered_graph, node_positions, positioned_graph
        # node_positions: list[tuple[key, x, y, cluster, degree or size]]
        # positioned_graph: str 内容为更新了节点的位置属性的graphml图
        # 节点的属性(key, type, description, source_id, cluster(本层有), level(本层有), degree,
        # human_readable_id, id, x, y, size)
        # 边的属性(source, target, weight, description, source_id, id, human_readable_id, level)
        {
            "id": "laid_out_entity_graph",
            "verb": "layout_graph",
            "args": {
                "embeddings_column": "embeddings",
                "graph_column": "clustered_graph",
                "to": "node_positions",
                "graph_to": "positioned_graph",
                **layout_graph_config,
            },
            "input": {"source": "workflow:create_base_entity_graph"},
        },
        # 将带位置信息的图的节点解包，每一行是一个节点，保留level列
        # 输入：level, clustered_graph, node_positions, positioned_graph
        # 输出：level, label, type, description, source_id, degree, human_readable_id, id, x, y,
        # size, graph_embedding, cluster
        {
            "verb": "unpack_graph",
            "args": {"column": "positioned_graph", "type": "nodes"},
        },
        # 丢掉节点的x, y属性
        # 输入：level, label, type, description, source_id, degree, human_readable_id, id, x, y,
        # size, graph_embedding, cluster
        # 输出：level, label, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, cluster
        {
            "id": "nodes_without_positions",
            "verb": "drop",
            "args": {"columns": ["x", "y"]},
        },
        *_compute_top_level_positions,
        # 将nodes_without_positions的表和_compute_top_level_node_positions的表在
        # id, top_level_node_id上合并
        # 输入：level, label, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, cluster
        # 输入: top_level_node_id, x, y
        # 输出：level, label, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, cluster, top_level_node_id, x, y
        {
            "verb": "join",
            "args": {
                "on": ["id", "top_level_node_id"],
            },
            "input": {
                "source": "nodes_without_positions",
                "others": ["_compute_top_level_node_positions"],
            },
        },
        # 将label和cluster改名
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
        # 输入：level, label, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, cluster, top_level_node_id, x, y
        # 输出：level, title, type, description, source_id, degree, human_readable_id, id,
        # size, graph_embedding, community, top_level_node_id, x, y
        {
            "verb": "rename",
            "args": {"columns": {"label": "title", "cluster": "community"}},
        },
    ]
