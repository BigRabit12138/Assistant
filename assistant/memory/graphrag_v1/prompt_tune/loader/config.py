from pathlib import Path

from assistant.memory.graphrag_v1.config import create_graphrag_config
from assistant.memory.graphrag_v1.index.progress.types import ProgressReporter


def read_config_parameters(
        root: str,
        reporter: ProgressReporter,
        config: str | None = None,
):
    _root = Path(root)
    settings_yaml = (
            Path(config)
            if config and Path(config).suffix in [".yaml", ".yml"]
            else _root / "settings.yaml"
    )
    if not settings_yaml.exists():
        settings_yaml = settings_yaml / "settings.yml"

    if settings_yaml.exists():
        reporter.info(f"Reading settings from {settings_yaml}")
        with settings_yaml.open("rb") as file:
            import yaml

            data = yaml.safe_load(file.read().decode(encoding="utf-8", errors="strict"))
            return create_graphrag_config(data, root)

    settings_json = (
        Path(config)
        if config and Path(config).suffix == ".json"
        else _root / "settings.json"
    )
    if settings_json.exists():
        reporter.info(f"Reading settings from {settings_json}")
        with settings_json.open("rb") as file:
            import json

            data = json.loads(file.read().decode(encoding="utf-8", errors="strict"))
            return create_graphrag_config(data, root)

    reporter.info("Reading settings from environment variables")
    return create_graphrag_config(root_dir=root)
