from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd

from src.core.types import Question


class BaseDatasetAdapter(ABC):
    """
    Base class for dataset adapters.
    Each dataset (GPQA, others) will:
      - load_raw: read raw/processed files into a DataFrame
      - to_questions: map rows to the standard Question schema
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._raw_df: Optional[pd.DataFrame] = None

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """
        Load the dataset into a pandas DataFrame.
        Should cache the result in self._raw_df.
        """
        raise NotImplementedError

    @abstractmethod
    def to_questions(self) -> List[Question]:
        """
        Convert the raw DataFrame into a list of Question objects.
        """
        raise NotImplementedError
