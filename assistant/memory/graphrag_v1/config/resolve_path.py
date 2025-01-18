import re

from pathlib import Path
from string import Template


def _resolve_timestamp_path_with_value(
        path: str | Path,
        timestamp_value: str
) -> Path:
    template = Template(str(path))
    resolved_path = template.substitute(timestamp=timestamp_value)
    return Path(resolved_path)


def _resolve_timestamp_path_with_dir(
        path: str | Path, pattern: re.Pattern[str]
) -> Path:
    path = Path(path)
    path_parts = path.parts
    parent_dir = Path(path_parts[0])
    found_timestamp_pattern = False
    for _, part in enumerate(path_parts[1:]):
        if part.lower() == "${timestamp}":
            found_timestamp_pattern = True
            break
        parent_dir = parent_dir / part

    if not found_timestamp_pattern:
        return path

    if not parent_dir.exists() or not parent_dir.is_dir():
        msg = f"Parent directory {parent_dir} does not exist or is not a directory."
        raise ValueError(msg)

    timestamp_dirs = [
        d for d in parent_dir.iterdir() if d.is_dir() and pattern.match(d.name)
    ]
    timestamp_dirs.sort(key=lambda d: d.name, reverse=True)
    if len(timestamp_dirs) == 0:
        msg = f"No timestamp directories found in {parent_dir} that match {pattern.pattern}."
        raise ValueError(msg)
    return _resolve_timestamp_path_with_value(path, timestamp_dirs[0].name)


def _resolve_timestamp_path(
        path: str | Path,
        pattern_or_timestamp_value: re.Pattern[str] | str | None = None,
) -> Path:
    if not pattern_or_timestamp_value:
        pattern_or_timestamp_value = re.compile(r"^\d{8}-\d{6}$")
    if isinstance(pattern_or_timestamp_value, str):
        return _resolve_timestamp_path_with_value(path, pattern_or_timestamp_value)
    return _resolve_timestamp_path_with_dir(path, pattern_or_timestamp_value)


def resolve_path(
        path_to_resolve: Path | str,
        root_dir: Path | str | None = None,
        pattern_or_timestamp_value: re.Pattern[str] | str | None = None,
) -> Path:
    if root_dir:
        path_to_resolve = (Path(root_dir) / path_to_resolve).resolve()
    else:
        path_to_resolve = Path(path_to_resolve)
    return _resolve_timestamp_path(path_to_resolve, pattern_or_timestamp_value)
