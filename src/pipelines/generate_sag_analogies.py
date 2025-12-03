import sys
from pathlib import Path
from typing import List
import os
import csv

# --- project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ensure directories
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "analogies"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# imports AFTER path hack
import src.models.langchain_chat  # noqa: F401  # ثبت مدل در رجیستری
import src.methods.sag  # noqa: F401           # ثبت متد در رجیستری

from src.core.registry import get_method
from src.data.loaders import load_questions_from_config
from src.core.types import Analogy
from src.models.factory import build_model_from_config


def save_analogies_to_csv(path: Path, analogies: List[Analogy]) -> None:
    fieldnames = [
        "analogy_id",
        "question_id",
        "method_name",
        "subject",
        "difficulty",
        "model_name",
        "model_id",
        "analogy_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in analogies:
            writer.writerow(
                {
                    "analogy_id": a.id,
                    "question_id": a.question_id,
                    "method_name": a.method_name,
                    "subject": a.metadata.get("subject"),
                    "difficulty": a.metadata.get("difficulty"),
                    "model_name": a.metadata.get("model_name"),
                    "model_id": a.metadata.get("model_id"),
                    "analogy_text": a.text,
                }
            )


def main():
    # --- ۱) لود سوال‌ها از GPQA
    dataset_config = {
        "version": "gpqa_main",
        "subject_filter": None,
        "subject_mode": "high_level",
        "difficulty_filter": None,
    }
    questions = load_questions_from_config("gpqa", dataset_config)
    print(f"[sag] Loaded {len(questions)} GPQA questions.")

    max_questions = int(os.getenv("SAG_DEBUG_N", "5"))
    questions = questions[:max_questions]
    print(f"[sag] Using first {len(questions)} questions for SAG generation.")

    # --- ۲) ساخت teacher از روی config
    model_config_path = PROJECT_ROOT / "config" / "models" / "teacher_gpt4_1_mini.yaml"
    teacher = build_model_from_config(model_config_path)
    print(f"[sag] Using teacher model: {teacher.name} ({teacher.model_id})")

    # --- ۳) ساخت متد SAG
    MethodCls = get_method("sag")
    sag_method = MethodCls(config={})

    # --- ۴) تولید آنالوژی‌ها
    analogies: List[Analogy] = []
    for idx, q in enumerate(questions):
        print(f"[sag] Generating analogy for question {idx+1}/{len(questions)} (id={q.id})")
        analogy = sag_method.run(q, teacher_model=teacher)
        analogies.append(analogy)

        preview = (analogy.text[:120] + "...") if len(analogy.text) > 120 else analogy.text
        print(f"[sag] Analogy preview: {preview}")

    # --- ۵) ذخیره‌ی CSV
    out_path = RESULTS_DIR / "gpqa_main_sag_sample.csv"
    save_analogies_to_csv(out_path, analogies)
    print(f"[sag] Saved {len(analogies)} analogies to {out_path}")


if __name__ == "__main__":
    main()
