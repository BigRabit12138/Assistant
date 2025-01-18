from typing_extensions import NotRequired, TypedDict


class ChunkingConfigInput(TypedDict):
    size: NotRequired[int | str | None]
    overlap: NotRequired[int | str | None]
    group_by_columns: NotRequired[list[str] | str | None]
    strategy: NotRequired[dict | None]
