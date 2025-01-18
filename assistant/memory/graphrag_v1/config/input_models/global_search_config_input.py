from typing_extensions import NotRequired, TypedDict


class GlobalSearchConfigInput(TypedDict):
    max_tokens: NotRequired[int | str | None]
    data_max_tokens: NotRequired[int | str | None]
    map_max_tokens: NotRequired[int | str | None]
    reduce_max_tokens: NotRequired[int | str | None]
    concurrency: NotRequired[int | str | None]
