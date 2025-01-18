from abc import ABC, abstractmethod
from typing import Dict, List, Union


class Parser(ABC):
    @abstractmethod
    def parse_aggregation_answer(
            self,
            states: List[Dict],
            texts: List[str],
    ) -> Union[Dict, List[Dict]]:
        pass

    @abstractmethod
    def parse_improve_answer(
            self,
            state: Dict,
            texts: List[str]
    ) -> Dict:
        pass

    @abstractmethod
    def parse_generate_answer(
            self,
            state: Dict,
            texts: List[str]
    ) -> List[Dict]:
        pass

    @abstractmethod
    def parse_validation_answer(
            self,
            state: Dict,
            texts: List[str]
    ) -> bool:
        pass

    @abstractmethod
    def parse_score_answer(
            self,
            states: List[Dict],
            texts: List[str],
    ) -> List[float]:
        pass
