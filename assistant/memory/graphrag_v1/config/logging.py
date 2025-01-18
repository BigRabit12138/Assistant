import logging

from pathlib import Path

from assistant.memory.graphrag_v1.config.enums import ReportingType
from assistant.memory.graphrag_v1.config.models.graph_rag_config import GraphRagConfig


def enable_logging(log_filepath: str | Path, verbose: bool = False) -> None:
    log_filepath = Path(log_filepath)
    log_filepath.parent.mkdir(parents=True, exist_ok=True)
    log_filepath.touch(exist_ok=True)

    logging.basicConfig(
        filename=log_filepath,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def enable_logging_with_config(
        config: GraphRagConfig, verbose: bool = False
) -> tuple[bool, str]:
    if config.reporting.type == ReportingType.file:
        log_path = Path(config.reporting.base_dir) / "indexing-engine.log"
        enable_logging(log_path, verbose)
        return True, str(log_path)
    return False, ""
