from typing import List, Optional, Literal
import pandas as pd

from src.core.types import Question
from .dataset_base import BaseDatasetAdapter


SubjectMode = Literal["high_level", "subdomain"]


class GPQADatasetAdapter(BaseDatasetAdapter):
    """
    Adapter برای GPQA.

    انتظار دارد CSV استانداردی که در prepare_gpqa ساخته‌ایم را بخواند، با ستون‌های:
      - question_uid
      - question
      - correct_answer
      - incorrect1/2/3
      - high_level_domain
      - subdomain
      - difficulty (اختیاری / None)
      - gpqa_version
    """

    def __init__(
        self,
        path: str,
        subject_filter: Optional[str] = None,
        subject_mode: SubjectMode = "high_level",
        difficulty_filter: Optional[List[str]] = None,
    ):
        super().__init__(path=path)
        self.subject_filter = subject_filter
        self.subject_mode = subject_mode
        self.difficulty_filter = difficulty_filter

    def load_raw(self) -> pd.DataFrame:
        if self._raw_df is None:
            df = pd.read_csv(self.path)

            # اعمال فیلتر موضوع
            if self.subject_filter is not None:
                if self.subject_mode == "high_level":
                    col = "high_level_domain"
                else:
                    col = "subdomain"
                if col in df.columns:
                    df = df[df[col] == self.subject_filter]
                else:
                    raise ValueError(
                        f"Expected column '{col}' not found in GPQA CSV."
                    )

            # اعمال فیلتر سختی (اگر بعداً ستونی برای سختی ایجاد شود)
            if self.difficulty_filter is not None and "difficulty" in df.columns:
                df = df[df["difficulty"].isin(self.difficulty_filter)]

            if "question_uid" not in df.columns:
                # fallback ساده در صورت غیبت
                df.insert(0, "question_uid", [f"gpqa_{i}" for i in range(len(df))])

            self._raw_df = df.reset_index(drop=True)

        return self._raw_df

    def to_questions(self) -> List[Question]:
        df = self.load_raw()
        questions: List[Question] = []

        for _, row in df.iterrows():
            # subject را بسته به mode انتخاب می‌کنیم
            if self.subject_mode == "high_level":
                subject_value = row.get("high_level_domain", "unknown")
            else:
                subject_value = row.get("subdomain", "unknown")

            # گزینه‌ها: correct + incorrect1/2/3
            options = []
            if "correct_answer" in df.columns:
                options.append(row["correct_answer"])
            for col in ["incorrect1", "incorrect2", "incorrect3"]:
                if col in df.columns and isinstance(row[col], str):
                    options.append(row[col])

            q = Question(
                id=str(row["question_uid"]),
                subject=str(subject_value),
                difficulty=row["difficulty"]
                if "difficulty" in df.columns
                else None,
                text=row["question"] if "question" in df.columns else "",
                options=options or None,
                correct_answer=row["correct_answer"]
                if "correct_answer" in df.columns
                else None,
            )
            questions.append(q)

        return questions
