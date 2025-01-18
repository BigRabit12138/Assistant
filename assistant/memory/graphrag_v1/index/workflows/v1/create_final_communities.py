from assistant.memory.graphrag_v1.index.config import (
    PipelineWorkflowStep,
    PipelineWorkflowConfig,
)

workflow_name = "create_final_communities"


def build_steps(
        _config: PipelineWorkflowConfig,
) -> list[PipelineWorkflowStep]:
    return [
        # 将图的节点解包，每一行是一个节点，保留level列
        # level: int 内容为层次
        # clustered_graph: str 内容为更新了本层节点的聚类属性的graphml图
        # 节点的属性(key, type, description, source_id, cluster(本层有), level(本层有), degree, human_readable_id, id)
        # 边的属性(source, target, weight, description, source_id, id, human_readable_id, level)
        # 输入：level, clustered_graph
        # 输出：level, label, type, description, source_id, degree, human_readable_id, id, graph_embedding,
        # cluster
        {
            "id": "graph_nodes",
            "verb": "unpack_graph",
            "args": {
                "column": "clustered_graph",
                "type": "nodes",
            },
            "input": {"source": "workflow:create_base_entity_graph"},
        },
        # 将图的边解包，每一行是一个边，保留level列
        # 输入：level, clustered_graph
        # 输出：level, source, target, weight, description, source_id, id, human_readable_id,
        {
            "id": "graph_edges",
            "verb": "unpack_graph",
            "args": {
                "column": "clustered_graph",
                "type": "edges",
            },
            "input": {"source": "workflow:create_base_entity_graph"},
        },
        # 将graph_nodes和graph_edges的表在label和source上合并
        # 输入：level, label, type, description, source_id, degree, human_readable_id, id, graph_embedding,
        # cluster
        # 输入：level, source, target, weight, description, source_id, id, human_readable_id,
        # 输出：level_1, label, type, description_1, source_id_1, degree, human_readable_id_1, id_1, graph_embedding,
        # cluster, level_2, source, target, weight, description_2,
        # source_id_2, id_2, human_readable_id_2,
        {
            "id": "source_clusters",
            "verb": "join",
            "args": {
                "on": ["label", "source"],
            },
            "input": {"source": "graph_nodes", "others": ["graph_edges"]},
        },
        # 将graph_nodes和graph_edges的表在label和target上合并
        # 输入：level, label, type, description, source_id, degree, human_readable_id, id, graph_embedding,
        # cluster
        # 输入：level, source, target, weight, description, source_id, id, human_readable_id,
        # 输出：level_1, label, type, description_1, source_id_1, degree, human_readable_id_1, id_1, graph_embedding,
        # cluster, level_2, source, target, weight, description_2,
        # source_id_2, id_2, human_readable_id_2,
        {
            "id": "target_clusters",
            "verb": "join",
            "args": {
                "on": ["label", "target"],
            },
            "input": {"source": "graph_nodes", "others": ["graph_edges"]},
        },
        # 将source_clusters和target_clusters的表的结果拼接成一个表
        # 输入：level_1, label, type, description_1, source_id_1, degree, human_readable_id_1, id_1, graph_embedding,
        # cluster, level_2, source, target, weight, description_2,
        # source_id_2, id_2, human_readable_id_2,
        # 输入：level_1, label, type, description_1, source_id_1, degree, human_readable_id_1, id_1, graph_embedding,
        # cluster, level_2, source, target, weight, description_2,
        # source_id_2, id_2, human_readable_id_2,
        # 输出：level_1, label, type, description_1, source_id_1, degree, human_readable_id_1, id_1, graph_embedding,
        # cluster, level_2, source, target, weight, description_2,
        # source_id_2, id_2, human_readable_id_2,
        {
            "id": "concatenated_clusters",
            "verb": "concat",
            "input": {
                "source": "source_clusters",
                "others": ["target_clusters"],
            },
        },
        # 保留处于节点和边处于同一层的行
        # 输入：level_1, label, type, description_1, source_id_1, degree, human_readable_id_1, id_1, graph_embedding,
        # cluster, level_2, source, target, weight, description_2,
        # source_id_2, id_2, human_readable_id_2,
        # 输出：level_1, label, type, description_1, source_id_1, degree, human_readable_id_1, id_1, graph_embedding,
        # cluster, level_2, source, target, weight, description_2,
        # source_id_2, id_2, human_readable_id_2,
        {
            "id": "combined_clusters",
            "verb": "filter",
            "args": {
                "column": "level_1",
                "criteria": [
                    {
                        "type": "column",
                        "operator": "equals",
                        "value": "level_2"
                    }
                ],
            },
            "input": {"source": "concatenated_clusters"},
        },
        # 按cluster, level_1分组，将id_2和source_id_1同一组的元素去除重复放入一个列表，并改名
        # 输入：level_1, label, type, description_1, source_id_1, degree, human_readable_id_1, id_1, graph_embedding,
        # cluster, level_2, source, target, weight, description_2,
        # source_id_2, id_2, human_readable_id_2,
        # 输出：cluster, level_1, relationship_ids, text_unit_ids
        {
            "id": "cluster_relationships",
            "verb": "aggregate_override",
            "args": {
                "groupby": [
                    "cluster",
                    "level_1",
                ],
                "aggregations": [
                    {
                        "column": "id_2",
                        "to": "relationship_ids",
                        "operation": "array_agg_distinct",
                    },
                    {
                        "column": "source_id_1",
                        "to": "text_unit_ids",
                        "operation": "array_agg_distinct",
                    },
                ],
            },
            "input": {"source": "combined_clusters"},
        },
        # 按cluster, level分组，将cluster一组的元素取第一个，并改名
        # 输入：level, label, type, description, source_id, degree, human_readable_id, id, graph_embedding,
        # cluster
        # 输出：cluster, level, id
        # id: int 内容是cluster
        {
            "id": "all_clusters",
            "verb": "aggregate_override",
            "args": {
                "groupby": ["cluster", "level"],
                "aggregations": [{"column": "cluster", "to": "id", "operation": "any"}],
            },
            "input": {"source": "graph_nodes"},
        },
        # 将all_clusters和cluster_relationships的表在id和cluster上合并
        # 输入：cluster, level, id
        # 输入：cluster, level_1, relationship_ids, text_unit_ids
        # 输出：cluster_1, level, id, cluster_2, level_1, relationship_ids, text_unit_ids
        {
            "verb": "join",
            "args": {
                "on": ["id", "cluster"],
            },
            "input": {"source": "all_clusters", "others": ["cluster_relationships"]},
        },
        # 保留level和level_1相同的行
        # 输入：cluster_1, level, id, cluster_2, level_1, relationship_ids, text_unit_ids
        # 输出：cluster_1, level, id, cluster_2, level_1, relationship_ids, text_unit_ids
        {
            "verb": "filter",
            "args": {
                "column": "level",
                "criteria": [
                    {
                        "type": "column",
                        "operator": "equals",
                        "value": "level_1"
                    }
                ],
            },
        },
        *create_community_title_wf,
        # 将id列赋值到raw_community
        # 输入：cluster_1, level, id, cluster_2, level_1, relationship_ids, text_unit_ids, __temp, title
        # 输出：cluster_1, level, id, cluster_2, level_1, relationship_ids, text_unit_ids, __temp, title, raw_community
        {
            "verb": "copy",
            "args": {
                "column": "id",
                "to": "raw_community",
            },
        },
        # 选择id, title, level, raw_community, relationship_ids, text_unit_ids列
        # 输入：cluster_1, level, id, cluster_2, level_1, relationship_ids, text_unit_ids, __temp, title, raw_community
        # 输出：id, title, level, raw_community, relationship_ids, text_unit_ids
        # id: str 聚簇ID
        # title: str 聚簇名称
        # level: int 聚簇的层
        # raw_community: str 聚簇ID
        # relationship_ids: 聚簇所包括的边
        # text_unit_ids： 聚簇所包括的边的实体的源chunk_id
        {
            "verb": "select",
            "args": {
                "columns": [
                    "id",
                    "title",
                    "level",
                    "raw_community",
                    "relationship_ids",
                    "text_unit_ids",
                ],
            },
        },
    ]


create_community_title_wf = [
    # 新建__temp列，并赋值Community
    # 输入：cluster_1, level, id, cluster_2, level_1, relationship_ids, text_unit_ids
    # 输出：cluster_1, level, id, cluster_2, level_1, relationship_ids, text_unit_ids, __temp
    {
        "verb": "fill",
        "args": {
            "to": "__temp",
            "value": "Community",
        },
    },
    # 将id, __temp合并，并赋值给title列
    # 输入：cluster_1, level, id, cluster_2, level_1, relationship_ids, text_unit_ids, __temp
    # 输出：cluster_1, level, id, cluster_2, level_1, relationship_ids, text_unit_ids, __temp, title
    {
        "verb": "merge",
        "args": {
            "columns": [
                "__temp",
                "id",
            ],
            "to": "title",
            "strategy": "concat",
            "preserveSource": True,
        },
    },
]
