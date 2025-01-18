from enum import Enum


class DocSelectionType(Enum):
    ALL = "all"
    RANDOM = "random"
    TOP = "top"
    AUTO = "auto"

    def __str__(self):
        return self.value
