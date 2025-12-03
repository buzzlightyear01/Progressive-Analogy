import sys
from pathlib import Path
from typing import Dict, List, Optional
import os
import csv

# --- project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ensure directories
ANSWERS_DIR = PROJECT_ROOT / "data" / "results" / "answers"
ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

# imports AFTER path hack
import src.models.langchain_chat  # noqa: F401  # رجیستر backend
import src.methods.sag  # noqa: F401  # برای آینده، اگر خواستیم reuse کنیم
import src.methods.pag  # noqa: F401

from src.data.loaders import load_questions_from_config
from src.models.factory import build_model_from_config
from src.core.types import Answer
from src.utils.config_loader import load_yaml


# ---------- کمکی: خواندن آنالوژی از CSV ----------

def load_analogies_csv(path: Path) -> Dict[str, str]:
    """
    CSV آنالوژی‌ها را می‌خواند و یک map از question_id -> analogy_text برمی‌گرداند.
    """
    if not path.exists():
        print(f"[student] WARNING: analogies file not found: {path}")
        return {}

    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["question_id"]
            text = row["analogy_text"]
            mapping[qid] = text
    print(f"[student] Loaded {len(mapping)} analogies from {path.name}")
    return mapping


# ---------- prompt template ها ----------

BASELINE_PROMPT = """You are a careful, step-by-step reasoner.

Answer the following multiple-choice question.
Explain your reasoning briefly, then clearly state your final answer choice.

QUESTION:
{question_text}

If you are unsure, choose the most plausible answer based on the reasoning.
"""


WITH_ANALOGY_PROMPT = """You are a careful, step-by-step reasoner.

You are given:
- A QUESTION
- An ANALOGY that was written to help understand the underlying concepts of the question.

First, read and internalize the ANALOGY.
Then reason step by step and answer the QUESTION.
Do NOT simply copy phrases from the analogy; use it to build your intuition.

ANALOGY:
{analogy_text}

QUESTION:
{question_text}

Explain your reasoning briefly, then clearly state your final answer choice.
"""


def build_student_prompt(
    question_text: str,
    analogy_text: Optional[str],
) -> str:
    if analogy_text is None:
        return BASELINE_PROMPT.format(question_text=question_text)
    else:
        return WITH_ANALOGY_PROMPT.format(
            question_text=question_text,
            analogy_text=analogy_text,
        )


# ---------- main pipeline ----------

def main():
    # --- ۱) لود config experiment
    exp_config_path = PROJECT_ROOT / "config" / "experiments" / "gpqa_student_basic.yaml"
    cfg = load_yaml(exp_config_path)

    dataset_cfg = cfg["dataset"]
    student_model_config_path = PROJECT_ROOT / cfg["student_model_config"]
    cond_cfgs = cfg["conditions"]

    # --- ۲) لود سوال‌ها
    questions = load_questions_from_config(
        dataset_cfg["name"],
        {
            "version": dataset_cfg.get("version", "gpqa_main"),
            "subject_filter": dataset_cfg.get("subject_filter"),
            "subject_mode": dataset_cfg.get("subject_mode", "high_level"),
            "difficulty_filter": dataset_cfg.get("difficulty_filter"),
        },
    )
    print(f"[student] Loaded {len(questions)} questions from dataset={dataset_cfg['name']}")

    max_questions = int(os.getenv("STUDENT_DEBUG_N", "5"))
    questions = questions[:max_questions]
    print(f"[student] Using first {len(questions)} questions for answering.")

    # --- ۳) آماده‌سازی map آنالوژی‌ها برای هر condition
    condition_analogies: Dict[str, Dict[str, str]] = {}
    for cond in cond_cfgs:
        name = cond["name"]
        ctype = cond["type"]
        if ctype == "with_analogy_csv":
            csv_rel_path = cond["analogy_csv"]
            csv_path = PROJECT_ROOT / csv_rel_path
            condition_analogies[name] = load_analogies_csv(csv_path)
        else:
            condition_analogies[name] = {}  # baseline یا نوع‌های دیگر که analogy خاص ندارند

    # --- ۴) ساخت student model از روی config
    student = build_model_from_config(student_model_config_path)
    print(f"[student] Using student model: {student.name} ({student.model_id})")

    # --- ۵) گرفتن جواب برای هر condition
    answers: List[Answer] = []

    for idx, q in enumerate(questions):
        print(f"[student] Question {idx+1}/{len(questions)} (id={q.id})")
        q_text = q.text  # متن سوال را برای prompt استفاده می‌کنیم

        for cond in cond_cfgs:
            cname = cond["name"]
            ctype = cond["type"]

            if ctype == "baseline":
                analogy_text = None
            elif ctype == "with_analogy_csv":
                analogy_map = condition_analogies.get(cname, {})
                analogy_text = analogy_map.get(q.id)
                if analogy_text is None:
                    print(f"[student]   [{cname}] No analogy found for question {q.id}, skipping.")
                    continue
            else:
                print(f"[student]   WARNING: unknown condition type '{ctype}' for '{cname}', skipping.")
                continue

            prompt = build_student_prompt(q_text, analogy_text)
            print(f"[student]   Condition={cname} → calling student model...")

            try:
                answer_text = student.generate(prompt, temperature=student.temperature)
            except Exception as e:
                print(f"[student]   ERROR calling student model: {e}")
                continue

            ans = Answer(
                question_id=q.id,
                student_model=student.name,
                method_name=cname,  # اینجا روش = نام condition
                answer_text=answer_text,
                metadata={
                    "subject": q.subject,
                    "difficulty": q.difficulty,
                    "model_id": student.model_id,
                    "condition_type": ctype,
                    "used_analogy": ctype == "with_analogy_csv",
                },
            )
            answers.append(ans)

    # --- ۶) ذخیره‌ی CSV خروجی
    out_path = ANSWERS_DIR / "gpqa_main_student_gpt_nano_5_modular.csv"
    fieldnames = [
        "question_id",
        "student_model",
        "condition",
        "subject",
        "difficulty",
        "model_id",
        "answer_text",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in answers:
            writer.writerow(
                {
                    "question_id": a.question_id,
                    "student_model": a.student_model,
                    "condition": a.method_name,
                    "subject": a.metadata.get("subject"),
                    "difficulty": a.metadata.get("difficulty"),
                    "model_id": a.metadata.get("model_id"),
                    "answer_text": a.answer_text,
                }
            )

    print(f"[student] Saved {len(answers)} answers to {out_path}")


if __name__ == "__main__":
    main()
