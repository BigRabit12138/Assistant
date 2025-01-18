from typing_extensions import NotRequired, TypedDict


class UmapConfigInput(TypedDict):
    enabled: NotRequired[bool | str | None]
