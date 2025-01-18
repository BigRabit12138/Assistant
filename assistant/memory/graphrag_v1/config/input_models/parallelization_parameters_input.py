from typing_extensions import NotRequired, TypedDict


class ParallelizationParametersInput(TypedDict):
    stagger: NotRequired[float | str | None]
    num_threads: NotRequired[int | str | None]
