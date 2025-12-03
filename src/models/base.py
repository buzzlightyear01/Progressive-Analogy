from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List


class LLMModel(ABC):
    """
    Abstract base class for all LLM wrappers (teacher, student, judge).
    Implementation details (LangChain, raw OpenAI, etc.) در کلاس‌های فرزند می‌آید.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        role: str = "generic",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ):
        self.name = name          # اسم منطقی داخل پروژه (مثلا "teacher_llama_7b")
        self.model_id = model_id  # اسم واقعی مدل برای API (مثلا "meta-llama/Meta-Llama-3-8B-Instruct")
        self.role = role          # "teacher" / "student" / "judge"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_config = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """
        Simple, blocking text generation interface.
        Concrete subclasses decide how to talk to the LLM backend.
        """
        raise NotImplementedError

    async def agenerate(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        """
        Optional async interface. By default, just calls generate in a thread.
        اگر خواستی بعدا نسخه‌های async با LangChain/asyncio بنویسی، این متد را override می‌کنی.
        """
        # برای سادگی فعلا روی نسخه‌ی sync می‌مانیم.
        from functools import partial
        import asyncio

        loop = asyncio.get_event_loop()
        func = partial(self.generate, prompt, **kwargs)
        return await loop.run_in_executor(None, func)
