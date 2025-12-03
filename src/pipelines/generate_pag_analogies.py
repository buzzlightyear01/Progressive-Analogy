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
import src.methods.pag  # noqa: F401            # ثبت متد PAG در رجیستری

from src.core.registry import get_method
from src.data.loaders import load_questions_from_config
from src.core.types import Analogy
from src.models.factory import build_model_from_config


def save_pag_analogies_to_csv(path: Path, analogies: List[Analogy]) -> None:
    fieldnames = [
        "analogy_id",
        "question_id",
        "method_name",
        "subject",
        "difficulty",
        "model_name",
        "model_id",
        "concept_steps",
        "analogy_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in analogies:
            concept_steps = a.metadata.get("concept_steps") or []
            concept_steps_str = " || ".join(concept_steps)
            writer.writerow(
                {
                    "analogy_id": a.id,
                    "question_id": a.question_id,
                    "method_name": a.method_name,
                    "subject": a.metadata.get("subject"),
                    "difficulty": a.metadata.get("difficulty"),
                    "model_name": a.metadata.get("model_name"),
                    "model_id": a.metadata.get("model_id"),
                    "concept_steps": concept_steps_str,
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
    print(f"[pag] Loaded {len(questions)} GPQA questions.")

    max_questions = int(os.getenv("PAG_DEBUG_N", "5"))
    questions = questions[:max_questions]
    print(f"[pag] Using first {len(questions)} questions for PAG generation.")

    # --- ۲) ساخت teacher از روی config
    model_config_path = PROJECT_ROOT / "config" / "models" / "teacher_gpt4_1_mini.yaml"
    teacher = build_model_from_config(model_config_path)
    print(f"[pag] Using teacher model: {teacher.name} ({teacher.model_id})")

    # --- ۳) ساخت متد PAG
    MethodCls = get_method("pag")
    pag_method = MethodCls(config={})

    # --- ۴) تولید آنالوژی‌ها
    analogies: List[Analogy] = []
    for idx, q in enumerate(questions):
        print(f"[pag] Generating analogy for question {idx+1}/{len(questions)} (id={q.id})")

        analogy = pag_method.run(q, teacher_model=teacher)
        analogies.append(analogy)

        preview = (analogy.text[:120] + "...") if len(analogy.text) > 120 else analogy.text
        print(f"[pag] Analogy preview: {preview}")

        steps = analogy.metadata.get("concept_steps") or []
        if steps:
            print("[pag] Concept steps (preview):")
            for s_idx, s in enumerate(steps[:3]):
                short = (s[:100] + "...") if len(s) > 100 else s
                print(f"   - {s_idx+1}: {short}")

    # --- ۵) ذخیره‌ی CSV
    out_path = RESULTS_DIR / "gpqa_main_pag_sample.csv"
    save_pag_analogies_to_csv(out_path, analogies)
    print(f"[pag] Saved {len(analogies)} PAG analogies to {out_path}")


if __name__ == "__main__":
    main()
