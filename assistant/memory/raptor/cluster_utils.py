import umap
import random
import logging
import tiktoken
import numpy as np

from typing import List, Optional
from abc import ABC, abstractmethod
from sklearn.mixture import GaussianMixture

from .utils import get_embeddings
from .tree_structures import Node

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        num_neighbors: int = 10,
        metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors,
        n_components=dim,
        metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
        embeddings: np.ndarray,
        max_clusters: int = 50,
        random_state: int = RANDOM_SEED
) -> np.ndarray:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def gmm_cluster(
        embeddings: np.ndarray,
        threshold: float,
        random_state: int = 0
):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
        verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) - 2))
    global_clusters, n_global_clusters = gmm_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}, {len(global_cluster_embeddings_)}"
            )

        if len(global_cluster_embeddings_) == 0:
            continue

        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = gmm_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")

    return all_local_clusters


class ClusteringAlgorithm(ABC):
    @staticmethod
    @abstractmethod
    def perform_clustering(
            nodes: List[Node],
            embedding_model_name: str,
            max_length_in_cluster: int = 3500,
            tokenizer=tiktoken.get_encoding("cl100k_base"),
            reduction_dimension: int = 10,
            threshold: float = 0.1,
            verbose: bool = False,
    ) -> List[List[Node]]:
        pass


class RaptorClustering(ClusteringAlgorithm):
    @staticmethod
    def perform_clustering(
            nodes: List[Node],
            embedding_model_name: str,
            max_length_in_cluster: int = 3500,
            tokenizer=tiktoken.get_encoding("cl100k_base"),
            reduction_dimension: int = 10,
            threshold: float = 0.1,
            verbose: bool = False,
    ) -> List[List[Node]]:
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])

        clusters = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        node_clusters = []

        for label in np.unique(np.concatenate(clusters)):
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            cluster_nodes = [nodes[i] for i in indices]

            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"reducing cluster with {len(cluster_nodes)} nodes"
                    )

                node_clusters.extend(
                    RaptorClustering.perform_clustering(
                        cluster_nodes, embedding_model_name, max_length_in_cluster
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
