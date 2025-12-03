from typing import Dict, Any
from src.core.types import Analogy, Judgment
from .metrics import default_metrics_schema

def judge_analogy(analogy: Analogy, judge_model_name: str, extra_context: Dict[str, Any]) -> Judgment:
    # TODO: implement LLM-as-judge call
    scores = default_metrics_schema()
    return Judgment(
        analogy_id=analogy.id,
        judge_model=judge_model_name,
        scores=scores,
        metadata=extra_context,
    )
