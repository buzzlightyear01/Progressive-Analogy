from typing import Any, Dict

from src.core.registry import register_method
from src.core.types import Question, Analogy
from src.methods.base import BaseMethod


SAG_DEFAULT_PROMPT = """You are an expert science tutor.

Your task is to create ONE analogy that helps a student intuitively understand the following QUESTION.
The analogy should:
- Capture the key underlying concept, not surface details.
- Use a familiar, everyday scenario.
- Avoid revealing the exact correct answer.
- Be concise (3–6 sentences).

QUESTION:
{question_text}

If answer options are provided, think about them silently but DO NOT mention them explicitly.
"""


@register_method("sag")
class SingleAnalogyGenerationMethod(BaseMethod):
    """
    Single Analogy Generation (SAG).

    Given a Question and a teacher LLM, produce ONE analogy as a free-form text.
    """

    name = "sag"

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config=config)
        # اجازه می‌دهی prompt در config override شود
        self.prompt_template: str = self.config.get(
            "prompt_template",
            SAG_DEFAULT_PROMPT,
        )

    def build_prompt(self, question: Question) -> str:
        """
        Build the prompt string for the teacher LLM based on the question.
        فعلاً ساده: فقط متن سوال؛ اگر خواستی بعداً گزینه‌ها یا متادیتا اضافه کنیم.
        """
        # ترجیحاً متن سوال فقط؛ اگر options خواستی اضافه کنی، بعداً در این بخش گسترش می‌دهیم.
        return self.prompt_template.format(
            question_text=question.text,
        )

    def run(self, question: Question, teacher_model: Any) -> Analogy:
        prompt = self.build_prompt(question)
        response_text = teacher_model.generate(prompt)

        analogy = Analogy(
            id=f"{self.name}_{question.id}",
            question_id=question.id,
            method_name=self.name,
            text=response_text,
            metadata={
                "prompt": prompt,
                "subject": question.subject,
                "difficulty": question.difficulty,
                "model_name": getattr(teacher_model, "name", None),
                "model_id": getattr(teacher_model, "model_id", None),
            },
        )
        return analogy
