import umap
import numpy as np
import networkx as nx
import graspologic as gc
import matplotlib.pyplot as plt

from assistant.memory.graphrag_v1.index.graph.visualization.typing import NodePosition


def get_zero_positions(
        node_labels: list[str],
        node_categories: list[int] | None = None,
        node_sizes: list[int] | None = None,
        three_d: bool | None = False,
) -> list[NodePosition]:
    """
    获取节点的空间布局，位置全部放置在原点
    :param node_labels: 节点
    :param node_categories: 节点的簇
    :param node_sizes: 节点的度
    :param three_d: 是否将位置布置在3维空间
    :return: 节点的布局
    """
    embedding_position_data: list[NodePosition] = []
    for index, node_name in enumerate(node_labels):
        # 获取当前节点的簇和度
        node_category = 1 if node_categories is None else node_categories[index]
        node_size = 1 if node_sizes is None else node_sizes[index]

        # 二维原点点布局
        if not three_d:
            embedding_position_data.append(
                NodePosition(
                    label=str(node_name),
                    x=0,
                    y=0,
                    cluster=str(int(node_category)),
                    size=int(node_size),
                )
            )
        else:
            embedding_position_data.append(
                NodePosition(
                    label=str(node_name),
                    x=0,
                    y=0,
                    z=0,
                    cluster=str(int(node_category)),
                    size=int(node_size),
                )
            )
    return embedding_position_data


def compute_umap_positions(
        embedding_vectors: np.ndarray,
        node_labels: list[str],
        node_categories: list[int] | None = None,
        node_sizes: list[int] | None = None,
        min_dist: float = 0.75,
        n_neighbors: int = 25,
        spread: int = 1,
        metric: str = "euclidean",
        n_components: int = 2,
        random_state: int = 86,
) -> list[NodePosition]:
    embedding_positions = umap.UMAP(
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        spread=spread,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    ).fit_transform(embedding_vectors)

    embedding_position_data: list[NodePosition] = []
    for index, node_name in enumerate(node_labels):
        node_points = embedding_positions[index]
        node_category = 1 if node_categories is None else node_categories[index]
        node_size = 1 if node_sizes is None else node_sizes[index]

        if len(node_points) == 2:
            embedding_position_data.append(
                NodePosition(
                    label=str(node_name),
                    x=float(node_points[0]),
                    y=float(node_points[1]),
                    z=float(node_points[2]),
                    cluster=str(int(node_category)),
                    size=int(node_size),
                )
            )
        return embedding_position_data


def visualize_embedding(
        graph,
        umap_positions: list[dict],
):
    plt.clf()
    figure = plt.gcf()
    ax = plt.gca()

    ax.set_axis_off()
    figure.set_size_inches(10, 10)
    figure.set_dpi(400)

    node_position_dict = {
        str(position["label"]): (position["x"], position["y"])
        for position in umap_positions
    }
    node_category_dict = {
        str(position["label"]): position["category"] for position in umap_positions
    }
    node_sizes = [position["size"] for position in umap_positions]
    node_colors = gc.layouts.categorical_colors(node_category_dict)

    vertices = []
    node_color_list = []
    for node in node_position_dict:
        vertices.append(node)
        node_color_list.append(node_colors[node])

    nx.draw_networkx_nodes(
        graph,
        pos=node_position_dict,
        nodelist=vertices,
        node_color=node_color_list,
        alpha=1.0,
        linewidths=0.01,
        node_size=node_sizes,
        node_shape='o',
        ax=ax
    )
    plt.show()
