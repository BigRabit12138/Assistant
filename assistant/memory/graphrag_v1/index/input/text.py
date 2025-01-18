import re
import logging

from typing import Any
from pathlib import Path

import pandas as pd

from assistant.memory.graphrag_v1.index.utils import gen_md5_hash
from assistant.memory.graphrag_v1.index.storage import PipelineStorage
from assistant.memory.graphrag_v1.index.progress import ProgressReporter
from assistant.memory.graphrag_v1.index.config import PipelineInputConfig

DEFAULT_FILE_PATTERN = re.compile(
    r".*[\\/](?P<source>[^\\/]+)[\\/](?P<year>\d{4})-(?P<month>\d{2})-"
    r"(?P<day>\d{2})_(?P<auther>[^_]+)_\d+\.txt"
)
input_type = "text"
log = logging.getLogger(__name__)


async def load(
        config: PipelineInputConfig,
        progress: ProgressReporter | None,
        storage: PipelineStorage,
) -> pd.DataFrame:

    async def load_file(
            path: str,
            group_: dict | None = None,
            _encoding: str = "utf-8"
    ) -> dict[str, Any]:
        if group_ is None:
            group_ = {}

        text = await storage.get(path, encoding="utf-8")
        new_item = {**group_, "text": text}
        new_item["id"] = gen_md5_hash(new_item, new_item.keys())
        new_item["title"] = str(Path(path).name)
        return new_item

    files = list(
        storage.find(
            re.compile(config.file_pattern),
            progress=progress,
            file_filter=config.file_filter
        )
    )
    if len(files) == 0:
        msg = f"No text files found in {config.base_dir}"
        raise ValueError(msg)

    found_files = f"Found text files from {config.base_dir}, found {files}."
    log.info(found_files)

    files_loaded = []
    for file, group in files:
        try:
            files_loaded.append(await load_file(file, group))
        except Exception:
            log.warning(f"Warning! Error loading file {file}. Skipping...")

    log.info(f"Found {len(files)} files, loading {len(files_loaded)}")

    return pd.DataFrame(files_loaded)
