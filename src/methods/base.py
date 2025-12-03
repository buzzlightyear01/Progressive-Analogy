from abc import ABC, abstractmethod
from typing import Any, Dict
from src.core.types import Question, Analogy

class BaseMethod(ABC):
    name: str

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def run(self, question: Question, teacher_model: Any) -> Analogy:
        """
        Generate an analogy (or other auxiliary info) for a given question.
        """
        ...
