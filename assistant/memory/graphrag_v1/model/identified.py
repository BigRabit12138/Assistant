from dataclasses import dataclass


@dataclass
class Identified:
    id: str

    short_id: str | None
