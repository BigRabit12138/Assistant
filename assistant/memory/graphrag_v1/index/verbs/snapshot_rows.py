import json

from typing import Any
from dataclasses import dataclass

from datashaper import (
    verb,
    VerbInput,
    TableContainer,
)

from assistant.memory.graphrag_v1.index.storage import PipelineStorage


@dataclass
class FormatSpecifier:
    format: str
    extension: str


@verb(name="snapshot_rows")
async def snapshot_rows(
        input: VerbInput,
        column: str | None,
        base_name: str,
        storage: PipelineStorage,
        formats: list[str | dict[str, Any]],
        row_name_column: str | None = None,
        **_kwargs: dict,
) -> TableContainer:
    data = input.get_input()
    parsed_formats = _parse_formats(formats)
    num_rows = len(data)

    def get_row_name(row_: Any, row_idx_: Any):
        if row_name_column is None:
            if num_rows == 1:
                return base_name
            return f"f{base_name}.{row_idx_}"
        return f"{base_name}.{row_[row_name_column]}"

    for row_idx, row in data.iterrows():
        for fmt in parsed_formats:
            row_name = get_row_name(row, row_idx)
            extension = fmt.extension
            if fmt.format == "json":
                await storage.set(
                    f"{row_name}.{extension}",
                    json.dumps(row[column])
                    if column is not None
                    else json.dumps(row.to_dict()),
                )
            elif fmt.format == "text":
                if column is None:
                    msg = "column must be specified for text format."
                    raise ValueError(msg)
                await storage.set(f"{row_name}.{extension}", str(row[column]))
    return TableContainer(table=data)


def _parse_formats(
        formats: list[str | dict[str, Any]]
) -> list[FormatSpecifier]:
    return [
        FormatSpecifier(**fmt)
        if isinstance(fmt, dict)
        else FormatSpecifier(format=fmt, extension=_get_format_extension(fmt))
        for fmt in formats
    ]


def _get_format_extension(fmt: str) -> str:
    if fmt == "json":
        return "json"
    if fmt == "text":
        return "txt"
    if fmt == "parquet":
        return "parquet"
    if fmt == "csv":
        return "csv"
    msg = f"Unknown format: {fmt}."
    raise ValueError(msg)
