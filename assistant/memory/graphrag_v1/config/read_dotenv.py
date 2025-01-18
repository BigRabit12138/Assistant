import os
import logging

from pathlib import Path

from dotenv import dotenv_values

log = logging.getLogger(__name__)


def read_dotenv(root: str) -> None:
    env_path = Path(root) / ".env"
    if env_path.exists():
        log.info("Loading pipeline .env file")
        env_config = dotenv_values(f"{env_path}")
        for key, value in env_config.items():
            if key not in os.environ:
                os.environ[key] = value or ""
    else:
        log.info(f"No .env file found at {root}")
