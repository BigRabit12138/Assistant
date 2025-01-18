import pickle
import logging

from threading import Lock
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor

from .tree_structures import Node, Tree
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .cluster_utils import ClusteringAlgorithm, RaptorClustering
from .utils import (distances_from_embeddings, get_embeddings, get_children,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
            self,
            reduction_dimension=10,
            clustering_algorithm=RaptorClustering,
            clustering_params={},
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
            self,
            current_level_nodes: Dict[int, Node],
            all_tree_nodes: Dict[int, Node],
            layer_to_nodes: Dict[int, List[Node]],
            use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Using Cluster TreeBuilder")
        next_node_index = len(all_tree_nodes)

        def process_cluster(
                cluster_,
                new_level_nodes_,
                next_node_index_,
                summarization_length_,
                lock_
        ):
            node_texts = get_text(cluster_)
            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length_
            )

            logging.info(
                f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, "
                f"Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            _, new_parent_node = self.create_node(
                next_node_index_, summarized_text, {node.index for node in cluster_}
            )
            with lock_:
                new_level_nodes_[next_node_index_] = new_parent_node

        for layer in range(self.num_layers):
            new_level_nodes = {}
            logging.info(f"Constructing Layer {layer}")
            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )

            lock = Lock()

            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)
            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes
