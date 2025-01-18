from typing_extensions import NotRequired, TypedDict


class EmbedGraphConfigInput(TypedDict):
    enabled: NotRequired[bool | str | None]
    num_walks: NotRequired[int | str | None]
    walk_length: NotRequired[int | str | None]
    window_size: NotRequired[int | str | None]
    iterations: NotRequired[int | str | None]
    random_seed: NotRequired[int | str | None]
    strategy: NotRequired[dict | None]
