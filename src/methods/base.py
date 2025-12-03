from abc import ABC, abstractmethod
from typing import Any, Dict
from src.core.types import Question, Analogy


class BaseMethod(ABC):
    """
    Base class for all auxiliary-information generation methods
    (PAG, SAG, Hint, ToT, etc.).
    """

    name: str

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def run(self, question: Question, teacher_model: Any) -> Analogy:
        """
        Generate an analogy (or other auxiliary info) for a given question
        using the provided teacher_model.
        """
        raise NotImplementedError
