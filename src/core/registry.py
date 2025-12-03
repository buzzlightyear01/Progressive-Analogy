from typing import Dict, Type, Any

_METHOD_REGISTRY: Dict[str, Any] = {}
_MODEL_REGISTRY: Dict[str, Any] = {}

def register_method(name: str):
    def decorator(cls):
        _METHOD_REGISTRY[name] = cls
        return cls
    return decorator

def get_method(name: str):
    return _METHOD_REGISTRY[name]

def register_model(name: str):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str):
    return _MODEL_REGISTRY[name]
