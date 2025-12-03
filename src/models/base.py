from abc import ABC, abstractmethod
from typing import Any, Dict

class LLMModel(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        ...
