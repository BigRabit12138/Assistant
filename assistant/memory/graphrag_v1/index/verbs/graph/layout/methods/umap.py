import logging
import traceback

from typing import Any

import numpy as np
import networkx as nx

from assistant.memory.graphrag_v1.index.typing import ErrorHandlerFn
from assistant.memory.graphrag_v1.index.verbs.graph.embed.typing import NodeEmbeddings
from assistant.memory.graphrag_v1.index.graph.visualization import (
    GraphLayout,
    NodePosition,
    compute_umap_positions,
)

log = logging.getLogger(__name__)


def run(
        graph: nx.Graph,
        embeddings: NodeEmbeddings,
        args: dict[str, Any],
        on_error: ErrorHandlerFn,
) -> GraphLayout:
    node_clusters = []
    node_sizes = []

    embeddings = _filter_raw_embeddings(embeddings)
    nodes = list(embeddings.keys())
    embedding_vectors = [embeddings[node_id] for node_id in nodes]

    for node_id in nodes:
        node = graph.nodes[node_id]
        cluster = node.get("cluster", node.get("community", -1))
        node_clusters.append(cluster)
        size = node.get("degree", node.get("size"), 0)
        node_sizes.append(size)

    additional_args = {}
    if len(node_clusters) > 0:
        additional_args["node_categories"] = node_clusters
    if len(node_sizes) > 0:
        additional_args["node_sizes"] = node_sizes

    try:
        return compute_umap_positions(
            embedding_vectors=np.array(embedding_vectors),
            node_labels=nodes,
            **additional_args,
            min_dist=args.get("min_dist", 0.75),
            n_neighbors=args.get("n_neighbors", 5),
        )
    except Exception as e:
        log.exception("Error running UMAP.")
        on_error(e, traceback.format_exc(), None)
        result = []
        for i in range(len(nodes)):
            cluster = node_clusters[i] if len(node_clusters) > 0 else 1
            result.append(
                NodePosition(
                    x=0,
                    y=0,
                    label=nodes[i],
                    size=0,
                    cluster=str(cluster),
                )
            )
        return result


def _filter_raw_embeddings(embeddings: NodeEmbeddings) -> NodeEmbeddings:
    return {
        node_id: embedding
        for node_id, embedding in embeddings.items()
        if embedding is not None
    }
