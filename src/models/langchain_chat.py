from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from src.core.registry import register_model
from src.models.base import LLMModel
from src.utils.env import get_env_or_raise, get_optional_env


OPENAI_API_KEY = ""  # اینجا کلید خودت رو بذار
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"          # یا اگر نخواستی، خالی بذار: ""
# =================================================================


@register_model("langchain_chat_openai")
class LangChainChatOpenAIModel(LLMModel):
    """
    Generic LangChain ChatOpenAI wrapper.

    Env vars / config:
      - می‌تونی مستقیم توی فایل، OPENAI_API_KEY و OPENAI_BASE_URL رو ست کنی
      - یا از env vars استفاده کنی:
        - OPENAI_API_KEY (یا OPENROUTER_API_KEY، اگر base_url = openrouter)
        - OPENAI_BASE_URL (اختیاری؛ مثلا https://openrouter.ai/api/v1)
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        role: str = "generic",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        # اسم env vars اگر خواستی هنوز به‌عنوان fallback استفاده می‌شن
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        # همچنین می‌تونی مستقیم توی کانفیگ این‌ها رو پاس بدی
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(
            name=name,
            model_id=model_id,
            role=role,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # اولویت:
        # 1) api_key / base_url پاس‌شده به constructor
        # 2) ثابت‌های بالای فایل (OPENAI_API_KEY / OPENAI_BASE_URL)
        # 3) env vars (OPENAI_API_KEY / OPENAI_BASE_URL)
        final_api_key = (
            api_key
            or (OPENAI_API_KEY or None)
            or get_env_or_raise(api_key_env)
        )
        final_base_url = (
            base_url
            or (OPENAI_BASE_URL or None)
            or get_optional_env(base_url_env, None)
        )

        client_kwargs: Dict[str, Any] = {
            "api_key": final_api_key,
            "model": model_id,
            "temperature": temperature,
        }
        if max_tokens is not None:
            client_kwargs["max_tokens"] = max_tokens
        if final_base_url:
            client_kwargs["base_url"] = final_base_url

        # LangChain ChatOpenAI client
        self._client = ChatOpenAI(**client_kwargs)

    def generate(self, prompt: str, **kwargs: Dict[str, Any]) -> str:
        temperature = kwargs.get("temperature", self.temperature)

        resp = self._client.invoke(prompt, temperature=temperature)
        return (getattr(resp, "content", "") or "").strip()


