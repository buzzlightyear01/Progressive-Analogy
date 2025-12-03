import sys
from pathlib import Path

# --- اضافه کردن روت پروژه به sys.path تا بتوانیم `import src...` انجام دهیم
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ❗ این خط خیلی مهم است: با این import، کلاس مدل لود می‌شود
# و دکوریتور @register_model اجرا می‌شود.
import src.models.langchain_chat  # noqa: F401

from src.core.registry import get_model, list_models


def main():
    print("[debug] Registered models:", list_models())
    ModelCls = get_model("langchain_chat_openai")

    teacher = ModelCls(
        name="teacher_llm",
        model_id="gpt-4.1-mini",   # این‌جا هر مدل سازگار با API خودت
        role="teacher",
        temperature=0.2,
        max_tokens=256,
    )

    prompt = "Say 'OK' if you are working."
    print("[debug] Sending prompt:", prompt)

    try:
        out = teacher.generate(prompt)
        print("[debug] Model output:", out)
    except Exception as e:
        print("[debug] Error while calling model:", e)


if __name__ == "__main__":
    main()
