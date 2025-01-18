from typing_extensions import NotRequired, TypedDict


class LocalSearchConfigInput(TypedDict):
    text_unit_prop: NotRequired[float | str | None]
    community_prop: NotRequired[float | str | None]
    conversation_history_max_turns: NotRequired[int | str | None]
    top_k_entities: NotRequired[int | str | None]
    top_k_relationships: NotRequired[int | str | None]
    max_tokens: NotRequired[int | str | None]
    llm_max_tokens: NotRequired[int | str | None]
