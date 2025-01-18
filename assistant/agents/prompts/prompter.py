from typing import Dict, List
from abc import ABC, abstractmethod


class Prompter(ABC):
    @abstractmethod
    def aggregation_prompt(
            self,
            state_dicts: List[Dict],
            **kwargs
    ) -> str:
        pass

    @abstractmethod
    def improve_prompt(
            self,
            **kwargs
    ) -> str:
        pass

    @abstractmethod
    def generate_prompt(
            self,
            num_branches: int,
            **kwargs
    ) -> str:
        pass

    @abstractmethod
    def validation_prompt(self, **kwargs) -> str:
        pass

    @abstractmethod
    def score_prompt(
            self,
            state_dicts: List[Dict],
            **kwargs
    ) -> str:
        pass

