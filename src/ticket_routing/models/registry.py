"""Predictor registry — adding a new model = `register("name")(cls)`.

No evaluator code or experiment-runner code changes when adding a predictor.
"""
from __future__ import annotations

from typing import Callable, Dict, Type

from .base import Predictor


_REGISTRY: Dict[str, Type[Predictor]] = {}


def register(name: str) -> Callable[[Type[Predictor]], Type[Predictor]]:
    def _wrap(cls: Type[Predictor]) -> Type[Predictor]:
        if name in _REGISTRY:
            raise ValueError(f"Predictor '{name}' already registered.")
        _REGISTRY[name] = cls
        return cls
    return _wrap


def build(name: str, **kwargs) -> Predictor:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown predictor '{name}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def registered_names() -> list[str]:
    return sorted(_REGISTRY)
