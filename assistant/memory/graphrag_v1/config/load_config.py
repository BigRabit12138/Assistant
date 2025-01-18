from pathlib import Path

from assistant.memory.graphrag_v1.config.config_file_loader import (
    load_config_from_file,
    search_for_config_in_root_dir,
)
from assistant.memory.graphrag_v1.config.create_graphrag_config import create_graphrag_config
from assistant.memory.graphrag_v1.config.enums import (
    StorageType,
    ReportingType,
)
from assistant.memory.graphrag_v1.config.models.graph_rag_config import GraphRagConfig
from assistant.memory.graphrag_v1.config.resolve_path import resolve_path


def load_config(
        root_dir: str | Path,
        config_filepath: str | Path | None = None,
        run_id: str | None = None,
) -> GraphRagConfig:
    root = Path(root_dir).resolve()
    if config_filepath:
        config_path = (root / config_filepath).resolve()
        if not config_path.exists():
            msg = f"Specified Config file not found: {config_path}"
            raise FileNotFoundError(msg)

    config_path = search_for_config_in_root_dir(root)
    if config_path:
        config = load_config_from_file(config_path)
    else:
        config = create_graphrag_config(root_dir=str(root))

    config.storage.base_dir = str(
        resolve_path(
            config.storage.base_dir,
            root if config.storage.type == StorageType.file else None,
            run_id,
        )
    )

    config.reporting.base_dir = str(
        resolve_path(
            config.reporting.base_dir,
            root if config.reporting.type == ReportingType.file else None,
            run_id,
        )
    )

    return config
