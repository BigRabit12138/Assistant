from dataclasses import dataclass


@dataclass
class Replacement:
    pattern: str
    replacement: str
