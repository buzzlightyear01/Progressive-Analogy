from typing import Any, Dict, List

from src.core.registry import register_method
from src.core.types import Question, Analogy
from src.methods.base import BaseMethod


CONCEPT_EXTRACTION_PROMPT = """You are an expert science educator.

Your task is to analyze the following QUESTION and extract the key conceptual steps
that a student needs to understand in order to solve it.

Write a numbered list of 3 to 6 conceptual steps.
Each step should be:
- about ONE core concept or relation
- phrased in a short, clear sentence
- ordered from most basic/foundational to most advanced/specific

QUESTION:
{question_text}

If answer options are present, silently consider them but DO NOT mention them.
OUTPUT FORMAT (numbered list):
1. ...
2. ...
3. ...
"""


PROGRESSIVE_ANALOGY_PROMPT = """You are an expert tutor who explains complex ideas using progressive analogies.

You are given:
- A QUESTION that the student needs to answer.
- A list of CONCEPT STEPS that represent the reasoning structure needed to solve the question.

Your task is to write ONE progressive analogy that:
- Has clearly separated stages/paragraphs that roughly follow the concept steps.
- Starts from a familiar, everyday scenario.
- Gradually introduces more detail/complexity, following the order of the concept steps.
- Helps the student build an intuitive mental model.
- Does NOT reveal the exact correct answer.
- Is around 5–10 sentences in total.

QUESTION:
{question_text}

CONCEPT STEPS:
{concept_steps_text}

Write the analogy as a short narrative, divided into logical steps or paragraphs, but do NOT explicitly say "Step 1, Step 2".
"""


def _parse_numbered_list(text: str) -> List[str]:
    """
    خیلی ساده: خطوطی که با رقم و نقطه شروع می‌شوند را به عنوان steps برمی‌گرداند.
    اگر نتوانیم تشخیص دهیم، کل متن را یک آیتم قرار می‌دهیم.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    steps: List[str] = []
    for line in lines:
        # مثال: "1. ..." یا "2) ..."
        if line[0].isdigit() and (line[1:3].startswith(".") or line[1:3].startswith(")")):
            # بعد از "1." یا "1)" را جدا کنیم
            rest = line[2:].lstrip(". ").lstrip(") ").strip()
            steps.append(rest if rest else line)
        else:
            # اگر قبلاً step داریم و این خط ادامه‌ی قبلی است، بچسبانیم
            if steps:
                steps[-1] += " " + line
            else:
                # هنوز step نداریم، شاید مدل بدون شماره‌گذاری نوشته
                steps.append(line)

    # اگر همه‌چیز خالی شد، fallback
    steps = [s for s in steps if s]
    if not steps:
        steps = [text.strip()]
    return steps


@register_method("pag")
class ProgressiveAnalogyGenerationMethod(BaseMethod):
    """
    Progressive Analogy Generation (PAG).

    دو call به teacher:
      1) استخراج concept steps
      2) ساخت progressive analogy با استفاده از آن steps
    """

    name = "pag"

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config=config)
        self.concept_prompt_template: str = self.config.get(
            "concept_prompt_template",
            CONCEPT_EXTRACTION_PROMPT,
        )
        self.analogy_prompt_template: str = self.config.get(
            "analogy_prompt_template",
            PROGRESSIVE_ANALOGY_PROMPT,
        )

    def build_concept_prompt(self, question: Question) -> str:
        return self.concept_prompt_template.format(
            question_text=question.text,
        )

    def build_analogy_prompt(self, question: Question, concept_steps: List[str]) -> str:
        concept_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(concept_steps))
        return self.analogy_prompt_template.format(
            question_text=question.text,
            concept_steps_text=concept_text,
        )

    def run(self, question: Question, teacher_model: Any) -> Analogy:
        # --- ۱) استخراج concept steps
        concept_prompt = self.build_concept_prompt(question)
        concept_raw = teacher_model.generate(concept_prompt)
        concept_steps = _parse_numbered_list(concept_raw)

        # --- ۲) ساخت progressive analogy
        analogy_prompt = self.build_analogy_prompt(question, concept_steps)
        analogy_text = teacher_model.generate(analogy_prompt)

        analogy = Analogy(
            id=f"{self.name}_{question.id}",
            question_id=question.id,
            method_name=self.name,
            text=analogy_text,
            metadata={
                "subject": question.subject,
                "difficulty": question.difficulty,
                "model_name": getattr(teacher_model, "name", None),
                "model_id": getattr(teacher_model, "model_id", None),
                "concept_prompt": concept_prompt,
                "concept_raw": concept_raw,
                "concept_steps": concept_steps,
                "analogy_prompt": analogy_prompt,
            },
        )
        return analogy
