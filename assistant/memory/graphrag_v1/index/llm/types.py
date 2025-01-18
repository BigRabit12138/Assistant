from typing import TypeAlias
from collections.abc import Callable

TextSplitter: TypeAlias = Callable[[str], list[str]]
TextListSplitter: TypeAlias = Callable[[list[str]], list[str]]
