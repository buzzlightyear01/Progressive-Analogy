from typing import List, Optional, Literal, Dict, Any

from src.core.types import Question
from .gpqa_adapter import GPQADatasetAdapter, SubjectMode
from .prepare_gpqa import ensure_gpqa_prepared

DatasetName = Literal["gpqa"]  # بعداً می‌تونی دیتاست‌های دیگر را اضافه کنی


def load_questions_from_config(
    dataset_name: DatasetName,
    config: Dict[str, Any],
) -> List[Question]:
    """
    High-level loader: با استفاده از یک نام دیتاست و یک config ساده،
    یک لیست unified از Question برمی‌گرداند.

    برای GPQA:
      config می‌تواند شامل چیزهایی مثل این باشد:
        - version: "gpqa_main" / "gpqa_extended" / "gpqa_diamond"
        - subject_filter: "Physics" / "Biology" / "Chemistry" / None
        - subject_mode: "high_level" (High-level domain) or "subdomain"
        - difficulty_filter: ["easy", "medium", ...] (اختیاری، برای آینده)
    """
    if dataset_name == "gpqa":
        version: str = config.get("version", "gpqa_main")
        subject_filter: Optional[str] = config.get("subject_filter")
        subject_mode: SubjectMode = config.get("subject_mode", "high_level")
        difficulty_filter = config.get("difficulty_filter")

        processed_csv_path = ensure_gpqa_prepared(version=version)

        adapter = GPQADatasetAdapter(
            path=str(processed_csv_path),
            subject_filter=subject_filter,
            subject_mode=subject_mode,
            difficulty_filter=difficulty_filter,
        )
        return adapter.to_questions()

    raise ValueError(f"Unknown dataset_name: {dataset_name}")
