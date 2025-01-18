import json

from pathlib import Path
from abc import ABC, abstractmethod

import yaml

from assistant.memory.graphrag_v1.config.create_graphrag_config import create_graphrag_config
from assistant.memory.graphrag_v1.config.models.graph_rag_config import GraphRagConfig

_default_config_files = ["settings.yaml", "settings.yml", "settings.json"]


def search_for_config_in_root_dir(root: str | Path) -> Path | None:
    root = Path(root)
    if not root.is_dir():
        msg = f"Invalid config path: {root} is not a directory"
        raise FileNotFoundError(msg)

    for file in _default_config_files:
        if (root / file).is_file():
            return root / file

    return None


class ConfigFileLoader(ABC):
    @abstractmethod
    def load_config(self, config_path: str | Path) -> GraphRagConfig:
        raise NotImplementedError


class ConfigYamlLoader(ConfigFileLoader):
    def load_config(self, config_path: str | Path) -> GraphRagConfig:
        config_path = Path(config_path)
        if config_path.suffix not in [".yaml", ".yml"]:
            msg = f"Invalid file extension for loading yaml config from: {config_path!s}. Expected .yaml or .yml"
            raise ValueError(msg)
        root_dir = str(config_path.parent)
        if not config_path.is_file():
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg)
        with config_path.open('rb') as file:
            data = yaml.safe_load(file.read().decode(encoding="utf-8", errors="strict"))
            return create_graphrag_config(data, root_dir)


class ConfigJsonLoader(ConfigFileLoader):
    def load_config(self, config_path: str | Path) -> GraphRagConfig:
        config_path = Path(config_path)
        root_dir = str(config_path.parent)
        if config_path.suffix != ".json":
            msg = f"Invalid file extension for loading json config from: {config_path!s}. Expected .json"
            raise ValueError(msg)
        if not config_path.is_file():
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg)
        with config_path.open("rb") as file:
            data = json.loads(file.read().decode(encoding="utf-8", errors="strict"))
            return create_graphrag_config(data, root_dir)


def get_config_file_loader(config_path: str | Path) -> ConfigFileLoader:
    config_path = Path(config_path)
    ext = config_path.suffix
    match ext:
        case ".yaml" | ".yml":
            return ConfigYamlLoader()
        case ".json":
            return ConfigJsonLoader()
        case _:
            msg = f"Unsupported config file extension: {ext}"
            raise ValueError(msg)


def load_config_from_file(config_path: str | Path) -> GraphRagConfig:
    loader = get_config_file_loader(config_path)
    return loader.load_config(config_path)
