from typing import Dict, Any

_METHOD_REGISTRY: Dict[str, Any] = {}
_MODEL_REGISTRY: Dict[str, Any] = {}


def register_method(name: str):
    def decorator(cls):
        _METHOD_REGISTRY[name] = cls
        return cls
    return decorator


def get_method(name: str):
    if name not in _METHOD_REGISTRY:
        raise KeyError(f"Method '{name}' is not registered.")
    return _METHOD_REGISTRY[name]


def register_model(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str):
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered.")
    return _MODEL_REGISTRY[name]


def list_models():
    return list(_MODEL_REGISTRY.keys())


def list_methods():
    return list(_METHOD_REGISTRY.keys())
