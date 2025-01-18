import os
import json
import logging

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any


class AbstractLanguageModel(ABC):
    def __init__(
            self,
            config_path: str = '',
            model_name: str = '',
            cache: bool = False
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: Dict | None = None
        self.model_name: str = model_name
        self.cache = cache
        if self.cache:
            self.response_cache: Dict[str, List[Any]] = {}

        self.load_config(config_path)
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cost: float = 0.0

    def load_config(
            self,
            path: str
    ) -> None:
        if path == '':
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, 'config.json')

        with open(path, 'r') as f:
            self.config = json.load(f)

        self.logger.debug(f'Loaded config from {path} for {self.model_name}')

    def clear_cache(self) -> None:
        self.response_cache.clear()

    @abstractmethod
    def query(
            self,
            query: str,
            num_responses: int = 1
    ) -> Any:
        pass

    @abstractmethod
    def get_response_texts(
            self,
            query_responses: Union[List[Any], Any]
    ) -> List[str]:
        pass
