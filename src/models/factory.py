from pathlib import Path
from typing import Any

from src.core.registry import get_model
from src.utils.config_loader import load_yaml


def build_model_from_config(config_path: str | Path) -> Any:
    """
    Read a YAML model config and instantiate the corresponding LLMModel.

    YAML example:

    backend: langchain_chat_openai
    name: teacher_gpt4_1_mini
    model_id: gpt-4.1-mini
    role: teacher
    temperature: 0.4
    max_tokens: 768
    api_key_env: OPENAI_API_KEY
    base_url_env: OPENAI_BASE_URL
    """
    cfg = load_yaml(config_path)
    backend = cfg.pop("backend", None)
    if backend is None:
        raise ValueError(f"Model config {config_path} is missing required key 'backend'.")

    ModelCls = get_model(backend)
    return ModelCls(**cfg)
