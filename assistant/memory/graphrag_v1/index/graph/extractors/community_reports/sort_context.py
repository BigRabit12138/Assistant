import pandas as pd

import assistant.memory.graphrag_v1.index.graph.extractors.community_reports.schemas as schemas

from assistant.memory.graphrag_v1.query.llm.text_utils import num_tokens


def sort_context(
        local_context: list[dict],
        sub_community_reports: list[dict] | None = None,
        max_tokens: int | None = None,
        node_id_column: str = schemas.NODE_ID,
        node_name_column: str = schemas.NODE_NAME,
        node_details_column: str = schemas.NODE_DETAILS,
        edge_id_column: str = schemas.EDGE_ID,
        edge_details_column: str = schemas.EDGE_DETAILS,
        edge_degree_column: str = schemas.EDGE_DEGREE,
        edge_source_column: str = schemas.EDGE_SOURCE,
        edge_target_column: str = schemas.EDGE_TARGET,
        claim_id_column: str = schemas.CLAIM_ID,
        claim_details_column: str = schemas.CLAIM_DETAILS,
        community_id_column: str = schemas.COMMUNITY_ID,
) -> str:
    """
    按边排序node_details, edge_details, claim_details，并获取对应的文本格式
    :param local_context: 所有对象的details
    :param sub_community_reports: 是否包括子聚簇
    :param max_tokens: 文本最大token限制
    :param node_id_column: 节点human_readable_id所在列
    :param node_name_column: 节点名字所在列
    :param node_details_column: 节点详细信息所在列
    :param edge_id_column: 边human_readable_id所在列
    :param edge_details_column: 边详细信息所在列
    :param edge_degree_column: 边的度所在列
    :param edge_source_column: 边的源点所在列
    :param edge_target_column: 边的目的点所在列
    :param claim_id_column: claim human_readable_id所在列
    :param claim_details_column: claim详细信息所在列
    :param community_id_column: 聚簇id所在列
    :return: 所有排序的对象的details的文本
    """
    def _get_context_string(
            entities: list[dict],
            edges_: list[dict],
            claims: list[dict],
            sub_community_reports_: list[dict] | None = None,
    ) -> str:
        """
        获取node_details, edge_details, claim_details文本格式
        :param entities: 节点的node_details
        :param edges_: 边的edge_details
        :param claims: claim的claim_details
        :param sub_community_reports_: 是否包括子聚簇
        :return: 所有对象details的文本格式
        """
        contexts = []
        # 默认不包含子community的信息汇集
        if sub_community_reports_:
            sub_community_reports_ = [
                report
                for report in sub_community_reports_
                if community_id_column in report
                and report[community_id_column]
                and str(report[community_id_column]).strip() != ""
            ]
            report_df = pd.DataFrame(sub_community_reports_).drop_duplicates()
            if not report_df.empty:
                if report_df[community_id_column].dtype == float:
                    report_df[community_id_column] = report_df[
                        community_id_column
                    ].astype(int)
                report_string = (
                    f"----Reports----\n{report_df.to_csv(index=False, sep=',')}"
                )
                contexts.append(report_string)
        # 获取human_readable_id不为空的节点
        entities = [
            entity
            for entity in entities
            if node_id_column in entity
            and entity[node_id_column]
            and str(entity[node_id_column]).strip() != ""
        ]
        # 去掉重复的节点
        entity_df = pd.DataFrame(entities).drop_duplicates()
        if not entity_df.empty:
            # 如果human_readable_id为float，转为int
            if entity_df[node_id_column].dtype == float:
                entity_df[node_id_column] = entity_df[node_id_column].astype(int)
            # 获取所有节点的文本格式
            entity_string = (
                f"-----Entities-----\n{entity_df.to_csv(index=False, sep=',')}"
            )
            contexts.append(entity_string)
        # 默认没有claims
        if claims and len(claims) > 0:
            claims = [
                claim
                for claim in claims
                if claim_id_column in claim
                and claim[claim_id_column]
                and str(claim[claim_id_column]).strip() != ""
            ]
            claim_df = pd.DataFrame(claims).drop_duplicates()
            if not claim_df.empty:
                if claim_df[claim_id_column].dtype == float:
                    claim_df[claim_id_column] = claim_df[claim_id_column].astype(int)
                claim_string = (
                    f"-----Claims-----\n{claim_df.to_csv(index=False, sep=',')}"
                )
                contexts.append(claim_string)

        # 获取human_readable_id不为空的边
        edges_ = [
            edge
            for edge in edges_
            if edge_id_column in edge
            and edge[edge_id_column]
            and str(edge[edge_id_column]).strip() != ""
        ]
        # 去掉重复的边
        edge_df = pd.DataFrame(edges_).drop_duplicates()
        if not edge_df.empty:
            # 如果human_readable_id为float，转为int
            if edge_df[edge_id_column].dtype == float:
                edge_df[edge_id_column] = edge_df[edge_id_column].astype(int)
            # 获取所有边的文本格式
            edge_string = (
                f"-----Relationships-----\n{edge_df.to_csv(index=False, sep=',')}"
            )
            contexts.append(edge_string)
        # 获取所有对象details的文本格式
        return "\n\n".join(contexts)

    edges = []
    node_details = {}
    claim_details = {}
    # 获取所有的edge_details, node_details, claim_details
    for record in local_context:
        node_name = record[node_name_column]
        record_edges = record.get(edge_details_column, [])
        record_edges = [e for e in record_edges if not pd.isna(e)]
        record_node_details = record[node_details_column]
        record_claims = record.get(claim_details_column, [])
        record_claims = [c for c in record_claims if not pd.isna(c)]

        edges.extend(record_edges)
        node_details[node_name] = record_node_details
        claim_details[node_name] = record_claims

    edges = [edge for edge in edges if isinstance(edge, dict)]
    # 按边的度倒排序
    edges = sorted(edges, key=lambda x: x[edge_degree_column], reverse=True)

    sorted_edges = []
    sorted_nodes = []
    sorted_claims = []
    context_string = ""
    # 按边，排序对应的node_details, claim_details
    for edge in edges:
        source_details = node_details.get(edge[edge_source_column], {})
        target_details = node_details.get(edge[edge_target_column], {})
        sorted_nodes.extend([source_details, target_details])
        sorted_edges.append(edge)
        source_claims = claim_details.get(edge[edge_source_column], [])
        target_claims = claim_details.get(edge[edge_target_column], [])
        sorted_claims.extend(source_claims if source_claims else [])
        sorted_claims.extend(target_claims if target_claims else [])
        # 如果有最大token限制，只收集指定数量的内容，默认没有
        if max_tokens:
            new_context_string = _get_context_string(
                sorted_nodes, sorted_edges, sorted_claims, sub_community_reports
            )
            if num_tokens(context_string) > max_tokens:
                break
            context_string = new_context_string

    if context_string == "":
        # 获取文本内容
        return _get_context_string(
            sorted_nodes, sorted_edges, sorted_claims, sub_community_reports
        )

    return context_string
