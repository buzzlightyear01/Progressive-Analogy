import os
from typing import Optional


def get_env_or_raise(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def get_optional_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)
