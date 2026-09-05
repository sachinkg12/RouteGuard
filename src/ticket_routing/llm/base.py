"""LLM client interface + response carrier.

Concrete clients (mock, openai_compatible, local_transformers) implement
`generate(prompt) -> LLMResponse`. The classifier never reaches behind this.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class LLMResponse:
    text: str
    token_logprobs: Optional[List[float]] = None
    raw: Any = None
    latency_seconds: float = 0.0
    cost_estimate: float = 0.0
    model: str = ""
    meta: dict = field(default_factory=dict)


class LLMClient(abc.ABC):
    kind: str = "abstract"
    model: str = "abstract"

    @abc.abstractmethod
    def generate(self, prompt: str) -> LLMResponse: ...


def build_client_from_config(cfg) -> LLMClient:
    """Factory dispatch; extend by adding a new branch + new client module."""
    kind = (cfg.kind or "mock").lower()
    if kind == "mock":
        from .mock_client import DryRunMockClient

        return DryRunMockClient(model=cfg.model)
    if kind == "openai_compatible":
        from .openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient(cfg)
    if kind == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(cfg)
    if kind == "local_transformers":
        from .local_transformers import LocalTransformersClient

        return LocalTransformersClient(cfg)
    raise ValueError(f"Unknown LLM client kind: {cfg.kind}")
