from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Question:
    id: str
    subject: str
    difficulty: Optional[str]
    text: str
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None

@dataclass
class Analogy:
    id: str
    question_id: str
    method_name: str
    text: str
    metadata: Dict[str, Any]

@dataclass
class Answer:
    question_id: str
    student_model: str
    method_name: str
    answer_text: str
    metadata: Dict[str, Any]

@dataclass
class Judgment:
    analogy_id: str
    judge_model: str
    scores: Dict[str, float]
    metadata: Dict[str, Any]
