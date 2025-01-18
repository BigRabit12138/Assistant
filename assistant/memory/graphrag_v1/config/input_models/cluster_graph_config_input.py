from typing_extensions import NotRequired, TypedDict


class ClusterGraphConfigInput(TypedDict):
    max_cluster_size: NotRequired[int | None]
    strategy: NotRequired[dict | None]
