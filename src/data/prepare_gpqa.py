from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd

from datasets import load_dataset
from huggingface_hub import login as hf_login


# پروژه:  src/data/prepare_gpqa.py
# روت پروژه = دو سطح بالاتر از این فایل
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "gpqa"
PROC_DIR = DATA_DIR / "processed"

###############################################
# def _maybe_login_to_hf() -> None:
#     """
#     اگر محیطت نیاز به توکن HuggingFace داشته باشه،
#     می‌تونی variable به اسم HF_TOKEN ست کنی و اینجا login می‌کنیم.
#     در غیر این صورت، load_dataset مستقیماً تلاش می‌کنه.
#     """
#     token = os.getenv("HF_TOKEN")
#     if token:
#         try:
#             hf_login(token=token)
#             print("[hf] Logged in to HuggingFace using HF_TOKEN.")
#         except Exception as e:
#             print(f"[hf] Warning: could not login with HF_TOKEN: {e}")

###############################################

def _maybe_login_to_hf() -> None:
    try:
        hf_login(
            token="",  
            add_to_git_credential=False
        )
        print("[hf] Logged in to HuggingFace using hardcoded token.")
    except Exception as e:
        print(f"[hf] Warning: could not login: {e}")

###############################################

def _standardize_gpqa_dataframe(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """
    df اولیه‌ی GPQA رو به یک DataFrame استاندارد برای خودمون تبدیل می‌کنیم.
    ستون‌های اصلی‌ای که استفاده می‌کنیم:
      - Question
      - Correct Answer
      - Incorrect Answer 1/2/3
      - High-level domain
      - Subdomain
    """
    # مطمئن شو ستون‌هایی که لازم داریم هستن
    needed_cols = [
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
        "High-level domain",
        "Subdomain",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"GPQA dataframe is missing expected columns: {missing}. "
            "Please inspect df.columns and update _standardize_gpqa_dataframe."
        )

    col_map: Dict[str, str] = {
        "Question": "question",
        "Correct Answer": "correct_answer",
        "Incorrect Answer 1": "incorrect1",
        "Incorrect Answer 2": "incorrect2",
        "Incorrect Answer 3": "incorrect3",
        "High-level domain": "high_level_domain",
        "Subdomain": "subdomain",
    }

    sdf = df[needed_cols].rename(columns=col_map).copy()

    # یک شناسه‌ی یکتا برای هر سوال
    sdf.insert(
        0,
        "question_uid",
        [f"{version}_{i}" for i in range(len(sdf))],
    )

    # جای خالی برای سختی (فعلاً GPQA چنین ستونی ندارد؛ بعداً می‌توانیم اضافه کنیم)
    if "difficulty" not in sdf.columns:
        sdf["difficulty"] = None

    # ورژن GPQA (main/diamond/extended)
    sdf["gpqa_version"] = version

    return sdf


def prepare_gpqa(version: str = "gpqa_main", force: bool = False) -> Path:
    """
    دیتا را از HuggingFace می‌گیرد، به فرمت استاندارد تبدیل می‌کند،
    و به صورت CSV در data/processed ذخیره می‌کند.

    خروجی: مسیر فایل CSV نهایی (مثلاً data/processed/gpqa_gpqa_main.csv)
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    processed_path = PROC_DIR / f"gpqa_{version}.csv"
    if processed_path.exists() and not force:
        print(f"[gpqa] Using existing processed file: {processed_path}")
        return processed_path

    print(f"[gpqa] Downloading GPQA version={version} from HuggingFace...")
    _maybe_login_to_hf()

    # NOTE: همان call نوت‌بوک: load_dataset("Idavidrein/gpqa", "gpqa_main")
    ds = load_dataset("Idavidrein/gpqa", version)
    train_split = ds["train"]
    df = train_split.to_pandas()

    # برای دیباگ محلی: می‌توانیم یک نسخه‌ی raw را هم ذخیره کنیم
    raw_path = RAW_DIR / f"gpqa_{version}_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"[gpqa] Saved raw CSV to {raw_path}")

    sdf = _standardize_gpqa_dataframe(df, version=version)
    sdf.to_csv(processed_path, index=False)
    print(f"[gpqa] Saved standardized CSV to {processed_path}")
    print(f"[gpqa] Total questions: {len(sdf)}")

    return processed_path


def ensure_gpqa_prepared(version: str = "gpqa_main") -> Path:
    """
    اگر فایل استاندارد موجود بود از همان استفاده می‌کند،
    وگرنه prepare_gpqa را صدا می‌زند.
    """
    return prepare_gpqa(version=version, force=False)


if __name__ == "__main__":
    path = prepare_gpqa("gpqa_main")
    print(f"[gpqa] Done. Processed file: {path}")
