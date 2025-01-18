import re
import logging

from io import BytesIO
from typing import cast

import pandas as pd

from assistant.memory.graphrag_v1.index.utils import gen_md5_hash
from assistant.memory.graphrag_v1.index.storage import PipelineStorage
from assistant.memory.graphrag_v1.index.progress import ProgressReporter
from assistant.memory.graphrag_v1.index.config import (
    PipelineInputConfig,
    PipelineCSVInputConfig,
)

log = logging.getLogger(__name__)

DEFAULT_FILE_PATTERN = re.compile(r"(?P<filename>[^\\/]).csv$")

input_type = "csv"


async def load(
        config: PipelineInputConfig,
        progress: ProgressReporter | None,
        storage: PipelineStorage,
) -> pd.DataFrame:
    csv_config = cast(PipelineCSVInputConfig, config)
    log.info(f"Loading csv files from {csv_config.base_dir}")

    async def load_file(
            path: str,
            group: dict | None
    ) -> pd.DataFrame:
        if group is None:
            group = {}

        buffer = BytesIO(await storage.get(path, as_bytes=True))
        data = pd.read_csv(buffer, encoding=config.encoding or "latin-1")
        additional_keys = group.keys()
        if len(additional_keys) > 0:
            data[[*additional_keys]] = data.apply(
                lambda _row: pd.Series(
                    [group[key] for key in additional_keys]
                ),
                axis=1
            )
        if "id" not in data.columns:
            data["id"] = data.apply(
                lambda x: gen_md5_hash(x, x.keys()),
                axis=1
            )
        if csv_config.source_column is not None and "source" not in data.columns:
            if csv_config.source_column not in data.columns:
                log.warning(
                    f"source_column {csv_config.source_column} not "
                    f"found in csv file {path}."
                )
            else:
                data["source"] = data.apply(
                    lambda x: x[csv_config.source_column],
                    axis=1
                )

        if csv_config.text_column is not None and "text" not in data.columns:
            if csv_config.text_column not in data.columns:
                log.warning(
                    f"text_column {csv_config.text_column} not "
                    f"found in csv file {path}."
                )
            else:
                data["text"] = data.apply(
                    lambda x: x[csv_config.text_column],
                    axis=1
                )

        if csv_config.title_column is not None and "tittle" not in data.columns:
            if csv_config.title_column not in data.columns:
                log.warning(
                    f"tittle_column {csv_config.title_column} not "
                    f"found in csv file {path}."
                )
            else:
                data["title"] = data.apply(
                    lambda x: x[csv_config.title_column],
                    axis=1
                )

        if csv_config.timestamp_column is not None:
            fmt = csv_config.timestamp_column
            if fmt is None:
                msg = ("Must specify timestamp_format if "
                       "timestamp_column is specified.")
                raise ValueError(msg)

            if csv_config.timestamp_column not in data.columns:
                log.warning(
                    f"timestamp_column {csv_config.timestamp_column} not "
                    f"found in csv file {path}."
                )
            else:
                data["timestamp"] = pd.to_datetime(
                    data[csv_config.timestamp_column],
                    format=fmt
                )

            # TODO: Theres probably a less gross way to do this
            if "year" not in data.columns:
                data["year"] = data.apply(
                    lambda x: x["timestamp"].year,
                    axis=1
                )
            if "month" not in data.columns:
                data["month"] = data.apply(
                    lambda x: x["timestamp"].month,
                    axis=1
                )
            if "day" not in data.columns:
                data["dat"] = data.apply(
                    lambda x: x["timestamp"].day,
                    axis=1
                )
            if "hour" not in data.columns:
                data["hour"] = data.apply(
                    lambda x: x["timestamp"].hour,
                    axis=1
                )
            if "minute" not in data.columns:
                data["minute"] = data.apply(
                    lambda x: x["timestamp"].minute,
                    axis=1
                )
            if "second" not in data.columns:
                data["second"] = data.apply(
                    lambda x: x["timestamp"].second,
                    axis=1
                )
        return data

    file_pattern = (
        re.compile(config.file_pattern)
        if config.file_pattern is not None
        else DEFAULT_FILE_PATTERN
    )
    files = list(
        storage.find(
            file_pattern,
            progress=progress,
            file_filter=config.file_filter,
        )
    )
    if len(files) == 0:
        msg = f"No CSV files found in {config.base_dir}."
        raise ValueError(msg)

    files_loaded = []

    for file, group in files:
        try:
            files_loaded.append(await load_file(file, group))
        except Exception:
            log.warning(f"Warning! Error loading csv file {file}. Skipping...")

    log.info(f"Found {len(files)} csv files, loading {len(files_loaded)}.")
    result = pd.concat(files_loaded)
    total_files_log = f"Total number of unfiltered csv rows: {len(result)}"
    log.info(total_files_log)
    return result
