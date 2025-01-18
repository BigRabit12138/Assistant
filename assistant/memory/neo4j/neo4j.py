import uuid
import time
import json
import pickle

from pathlib import Path
from datetime import (
    date,
    datetime,
)
from datetime import time as time_type

import py2neo
import pandas as pd
import numpy as np

from networkx import Graph


class Neo4j:
    def __init__(
            self,
            url="bolt://localhost:7687",
            username='neo4j',
            passwd='neo4jneo4j'
    ):
        try:
            self.neo4j_base = py2neo.Graph(url, auth=(username, passwd))
            self.node_matcher = py2neo.NodeMatcher(self.neo4j_base)
            self.relation_matcher = py2neo.RelationshipMatcher(self.neo4j_base)

        except Exception as e:
            print(f"Neo4j connection failed:\n{str(e)}")

    def is_valid_property(self, value):
        """
        检查一个值是否是合法的 Neo4j 属性值。
        """
        if value is None:
            return True
        elif isinstance(value, (int, float, bool, str)):
            return True
        elif isinstance(value, (date, time_type, datetime)):
            return True
        elif isinstance(value, (list, tuple)):
            return all(self.is_valid_property(v) for v in value)
        return False

    @staticmethod
    def is_ndarray(value):
        if isinstance(value, np.ndarray):
            return True
        return False

    @staticmethod
    def merge_properties(properties: list[dict]):
        new_property = dict()
        for node_property in properties:
            new_property.update(node_property)
        return new_property

    def trans_to_valid_property(self, node_property: dict):
        for key, value in node_property.items():
            if self.is_ndarray(value):
                value = value.tolist()
                node_property[key] = value
            if not self.is_valid_property(value):
                node_property[key] = json.dumps(value)

    def create_node(
            self,
            node_property: dict,
            node_id: str | None = None
    ):
        self.trans_to_valid_property(node_property)
        if node_id:
            node = self.query_node(node_id)
        else:
            node = None
        if node is None:
            node_id = str(uuid.uuid4()) if not node_id else node_id
            node = py2neo.Node(
                node_id,
                **{
                    **node_property,
                    "_create_count": 1,
                    "_last_create_time": time.time(),
                    "_read_count": 0,
                    "_last_read_time": 0.0,
                    "_write_count": 0,
                    "_last_write_time": 0.0,
                }
            )
            self.neo4j_base.create(node)
        else:
            print(f"node_id({node_id}): 已经存在！")
            node.update(node_property)
            node["_create_count"] += 1
            node["_last_create_time"] = time.time()
            self.neo4j_base.push(node)

        return {node_id: node_property}

    def create_relation(
            self,
            node_start: py2neo.Node,
            relation_property: dict,
            node_end: py2neo.Node,
            relation_id: str | None = None,
    ):
        self.trans_to_valid_property(relation_property)

        base_relation = self.query_relation_by_properties(
            node_start=node_start,
            relation=relation_id,
            node_end=node_end,
        )
        if base_relation:
            relation_id = type(base_relation[0]).__name__
            print(
                f"{node_start.labels}->{relation_id}->{node_end.labels}：已经存在！"
            )
            if len(base_relation) > 1:
                print(
                    f"{len(base_relation)}: {node_start.labels}->{relation_id}->{node_end.labels}"
                )
            base_relation = base_relation[0]
            base_relation.update(relation_property)
            base_relation["_create_count"] += 1
            base_relation["_last_create_time"] = time.time()
            self.neo4j_base.push(base_relation)
        else:
            relation_id = str(uuid.uuid4()) if not relation_id else relation_id
            relation = py2neo.Relationship(
                node_start,
                relation_id,
                node_end,
                **{
                    **relation_property,
                    "_create_count": 1,
                    "_last_create_time": time.time(),
                    "_read_count": 0,
                    "_last_read_time": 0.0,
                    "_write_count": 0,
                    "_last_write_time": 0.0,
                }
            )
            self.neo4j_base.create(relation)

        return {relation_id: relation_property}

    def add_graph_from_dataframe(
            self,
            input_dataframe: pd.DataFrame,
            is_real_graph: bool = True,
            is_node: bool = True,
            entity_workflow_name: str = "",
    ):
        """
        从表格中创建图
        :param entity_workflow_name: 生成节点的workflow的名字
        :param is_node: pandas表格是否是node
        :param is_real_graph: pandas表格是否是图
        :param input_dataframe: 输入表格
        :return:
        """
        if is_real_graph:
            file_path = Path('temp_graph_entity.pkl')
            if not file_path.exists():
                file_path.touch()
                with open(file_path, "wb") as file:
                    pickle.dump({}, file)
            with open(file_path, "rb") as file:
                temp_graph_entity = pickle.load(file)
            if is_node:
                all_nodes = []
                for index, row in input_dataframe.iterrows():
                    all_nodes.append(self.create_node(dict(row)))
                temp_graph_entity[entity_workflow_name] = all_nodes
                with open(file_path, "wb") as file:
                    file.write(pickle.dumps(temp_graph_entity))
            else:
                assert temp_graph_entity is not None, "请先添加节点，再添加关系！"
                for index, row in input_dataframe.iterrows():
                    node_start_id = self.get_node_id_by_name(
                        row['source'], entity_workflow_name
                    )
                    node_start = self.query_node(node_start_id)
                    node_end_id = self.get_node_id_by_name(
                        row['target'], entity_workflow_name
                    )
                    node_end = self.query_node(node_end_id)
                    self.create_relation(node_start, dict(row), node_end)
        else:
            last_node = None
            current_node = None
            for index, row in input_dataframe.iterrows():
                node_id = next(iter(self.create_node(dict(row)).keys()))
                node = self.query_node(node_id)
                current_node = node
                if last_node is not None and current_node is not None:
                    self.create_relation(last_node, {}, current_node)
                last_node = current_node

    def add_graph_from_networkx(
            self,
            networkx_graph: Graph,
    ):
        all_nodes = []
        for node, data in networkx_graph.nodes(data=True):
            all_nodes.append(
                self.create_node({'label': node, **data})
            )

        for u, v, data in networkx_graph.edges(data=True):
            node_start_id = self.get_node_id_by_name(u, all_nodes=all_nodes)
            node_start = self.query_node(node_start_id)
            node_end_id = self.get_node_id_by_name(v, all_nodes=all_nodes)
            node_end = self.query_node(node_end_id)
            self.create_relation(
                node_start,
                {"source": u, "target": v, **data},
                node_end
            )

    @staticmethod
    def get_node_id_by_name(
            node_name: str,
            entity_workflow_name: str | None = None,
            all_nodes: list | None = None,
    ):
        if entity_workflow_name is not None:
            file_path = Path('temp_graph_entity.pkl')
            assert file_path.exists(), "请先添加节点，再进行操作！"
            with open(file_path, "rb") as file:
                temp_graph_entity = pickle.load(file)
            all_nodes = temp_graph_entity.get(entity_workflow_name)
        assert all_nodes is not None, "请先添加节点，再进行操作！"
        assert all_nodes is not [], f"{entity_workflow_name}无节点！"

        for node in all_nodes:
            node_id = next(iter(node.keys()))
            node_property = next(iter(node.values()))
            possible_entity_name = (
                node_property.get("label"),
                node_property.get("entity_name"),
                node_property.get("entity"),
                node_property.get("title"),
            )
            if node_name in possible_entity_name:
                return node_id
        raise ValueError(f"{entity_workflow_name}不存在{node_name}")

    def query_node(self, node_id: str):
        node = self.node_matcher.match(node_id).first()
        if node is None:
            return None
        node['_read_count'] += 1
        node['_last_read_time'] = time.time()
        self.neo4j_base.push(node)
        return node

    def query_node_by_property_name(
            self,
            property_name: str,
            property_value: str,
    ):
        assert isinstance(property_name, str)

        nodes = self.node_matcher.match(
            **{property_name: property_value}
        ).all()
        if nodes is None:
            return None
        for node in nodes:
            node['_read_count'] += 1
            node['_last_read_time'] = time.time()
            self.neo4j_base.push(node)
        return nodes

    def query_relation_by_properties(
            self,
            node_start,
            relation,
            node_end
    ):
        relation_ships = self.relation_matcher.match(
            [node_start, node_end], r_type=relation
        ).all()
        if relation_ships is None:
            return None
        for relation_ship in relation_ships:
            relation_ship['_read_count'] += 1
            relation_ship['_last_read_time'] = time.time()
            self.neo4j_base.push(relation_ship)
        return relation_ships

    def query_relation(self, relationship_id):
        relationship = self.relation_matcher.match(
            r_type=relationship_id
        ).first()
        if relationship is None:
            return None
        relationship['_read_count'] += 1
        relationship['_last_read_time'] = time.time()
        self.neo4j_base.push(relationship)
        return relationship

    def add_graph_from_graphrag(
            self,
            graphrag_output_dir: str,
    ):
        final_entities = pd.read_parquet(
            Path(graphrag_output_dir) / "create_final_entities.parquet"
        )
        entities_id = final_entities['id']
        entities_id = list(entities_id)
        entities_id = set(entities_id)
        final_nodes = pd.read_parquet(
            Path(graphrag_output_dir) / "create_final_nodes.parquet"
        )
        nodes_id = final_nodes['id']
        nodes_id = list(nodes_id)
        nodes_id = set(nodes_id)
        nodes_id.union(entities_id)
        final_entities.set_index('id', inplace=True)
        final_nodes.set_index('id', inplace=True)
        for node_id in nodes_id:
            properties = [
                final_entities.loc[node_id].to_dict(),
                final_nodes.loc[node_id].to_dict(),
            ]
            node_property = self.merge_properties(properties=properties)
            self.create_node(node_property, node_id)

        final_relationships = pd.read_parquet(
            Path(graphrag_output_dir) / "create_final_relationships.parquet"
        )
        final_relationships_id = final_relationships['id']
        final_relationships_id = list(final_relationships_id)
        final_relationships_id = set(final_relationships_id)
        final_relationships.set_index('id', inplace=True)
        for relationship_id in final_relationships_id:
            properties = [
                final_relationships.loc[relationship_id].to_dict()
            ]
            relation_property = self.merge_properties(properties=properties)
            self.create_node(
                node_property=relation_property,
                node_id=relationship_id
            )

        final_communities = pd.read_parquet(
            Path(graphrag_output_dir) / "create_final_communities.parquet"
        )
        final_community_reports = pd.read_parquet(
            Path(graphrag_output_dir) / "create_final_community_reports.parquet"
        )
        final_community_reports_id = final_community_reports['id']
        final_community_reports_id = list(final_community_reports_id)
        final_community_reports_id = set(final_community_reports_id)
        final_communities.set_index('id', inplace=True)
        final_community_reports.set_index('id', inplace=True)
        for report_id in final_community_reports_id:
            report = final_community_reports.loc[report_id].to_dict()
            properties = [
                final_communities.loc[report['community']].to_dict(),
                report,
            ]
            node_property = self.merge_properties(properties=properties)
            self.create_node(node_property, report_id)

        final_text_units = pd.read_parquet(
            Path(graphrag_output_dir) / "create_final_text_units.parquet"
        )
        final_text_units_id = final_text_units['id']
        final_text_units_id = list(final_text_units_id)
        final_text_units_id = set(final_text_units_id)
        final_text_units.set_index('id', inplace=True)
        for unit_id in final_text_units_id:
            properties = [
                final_text_units.loc[unit_id].to_dict()
            ]
            node_property = self.merge_properties(properties=properties)
            self.create_node(node_property, unit_id)

        final_docs = pd.read_parquet(
            Path(graphrag_output_dir) / "create_final_documents.parquet"
        )
        final_docs_id = final_docs['id']
        final_docs_id = list(final_docs_id)
        final_docs_id = set(final_docs_id)
        final_docs.set_index('id', inplace=True)
        for doc_id in final_docs_id:
            properties = [
                final_docs.loc[doc_id].to_dict()
            ]
            node_property = self.merge_properties(properties=properties)
            self.create_node(node_property, doc_id)

        for index, row in final_docs.iterrows():
            doc_node = self.query_node(str(index))
            for text_unit_id in row['text_unit_ids']:
                text_unit_node = self.query_node(text_unit_id)
                self.create_relation(
                    node_start=doc_node,
                    relation_property={},
                    node_end=text_unit_node,
                )

        for index, row in final_text_units.iterrows():
            text_unit_node = self.query_node(str(index))
            for doc_id in row['document_ids']:
                doc_node = self.query_node(doc_id)
                self.create_relation(
                    node_start=text_unit_node,
                    relation_property={},
                    node_end=doc_node,
                )
            for entity_id in row['entity_ids']:
                entity_node = self.query_node(entity_id)
                self.create_relation(
                    node_start=text_unit_node,
                    relation_property={},
                    node_end=entity_node,
                )
            if row['relationship_ids'] is not None:
                for relationship_id in row['relationship_ids']:
                    relationship_node = self.query_node(relationship_id)
                    self.create_relation(
                        node_start=text_unit_node,
                        relation_property={},
                        node_end=relationship_node,
                    )

        for index in final_community_reports_id:
            community_node = self.query_node(str(index))
            text_unit_ids = community_node["text_unit_ids"]
            new_text_unit_ids = []
            for text_unit_id in text_unit_ids:
                if ',' in text_unit_id:
                    new_text_unit_ids.extend(
                        [unit_id.strip() for unit_id in text_unit_id.split(',') if unit_id.strip() != '']
                    )
                else:
                    new_text_unit_ids.append(text_unit_id)
            text_unit_ids = set(new_text_unit_ids)

            for text_unit_id in text_unit_ids:
                text_unit_node = self.query_node(text_unit_id)
                self.create_relation(
                    node_start=community_node,
                    relation_property={},
                    node_end=text_unit_node,
                )
            for relationship_id in community_node['relationship_ids']:
                relationship_node = self.query_node(relationship_id)

                self.create_relation(
                    node_start=community_node,
                    relation_property={},
                    node_end=relationship_node
                )

        for relationship_id in final_relationships_id:
            properties = [
                final_relationships.loc[relationship_id].to_dict()
            ]
            relation_property = self.merge_properties(properties=properties)
            node_start = self.query_node_by_property_name(
                property_name='title',
                property_value=final_relationships.loc[relationship_id]['source']
            )[0]
            node_end = self.query_node_by_property_name(
                property_name='title',
                property_value=final_relationships.loc[relationship_id]['target']
            )[0]
            relationship_node = self.query_node(relationship_id)

            self.create_relation(
                node_start=node_start,
                relation_property=relation_property,
                node_end=node_end,
                relation_id=relationship_id
            )
            self.create_relation(
                node_start=relationship_node,
                relation_property={},
                node_end=node_start,
            )
            self.create_relation(
                node_start=relationship_node,
                relation_property={},
                node_end=node_end
            )
            for text_unit_id in relationship_node['text_unit_ids']:
                text_unit_node = self.query_node(text_unit_id)
                self.create_relation(
                    node_start=relationship_node,
                    relation_property={},
                    node_end=text_unit_node,
                )

        for index in nodes_id:
            entity_node = self.query_node(str(index))
            for text_unit_id in entity_node["text_unit_ids"]:
                text_unit_node = self.query_node(text_unit_id)
                self.create_relation(
                    node_start=entity_node,
                    relation_property={},
                    node_end=text_unit_node,
                )
            if entity_node['community']:
                community_node = self.query_node_by_property_name(
                    property_name='community',
                    property_value=entity_node['community']
                )[0]
                self.create_relation(
                    node_start=entity_node,
                    relation_property={},
                    node_end=community_node,
                )

    def delete_relation(self, relation_ship):
        assert relation_ship is not None

        self.neo4j_base.separate(relation_ship)

    # def merge(self, node1, node2):
    #     assert type(node1) == py2neo.Node
    #     assert type(node2) == py2neo.Node
    #
    #     self.neo4j_base.merge(node1, node2)

    def sleep(self):
        nodes = self.neo4j_base.nodes
        for id_of_node in nodes:
            node = nodes[id_of_node]
            if node['count'] <= 0:
                self.neo4j_base.delete(node)
            else:
                node['count'] = node['count'] - 1
                self.neo4j_base.push(node)

        relations = self.neo4j_base.relationships
        for id_of_relation in relations:
            relation = relations[id_of_relation]
            if relation['count'] <= 0:
                self.neo4j_base.separate(relation)
            else:
                relation['count'] = relation['count'] - 1
                self.neo4j_base.push(relation)


neo4j_driver = Neo4j()
neo4j_driver.add_graph_from_graphrag(
    graphrag_output_dir="/home/wuzhenglin/PycharmProjects/Assistant/ragtest/output/20241004-165816/artifacts"
)
pass