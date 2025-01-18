from dataclasses import dataclass

import numpy as np
import networkx as nx
import graspologic as gc


@dataclass
class NodeEmbeddings:
    nodes: list[str]
    embeddings: np.ndarray


def embed_nod2vec(
        graph: nx.Graph | nx.DiGraph,
        dimensions: int = 1536,
        num_walks: int = 10,
        walk_length: int = 40,
        window_size: int = 2,
        iterations: int = 3,
        random_seed: int = 86,
) -> NodeEmbeddings:
    lcc_tensors = gc.embed.node2vec_embed(
        graph=graph,
        dimensions=dimensions,
        window_size=window_size,
        iterations=iterations,
        num_walks=num_walks,
        walk_length=walk_length,
        random_seed=random_seed,
    )
    return NodeEmbeddings(embeddings=lcc_tensors[0], nodes=lcc_tensors[1])
