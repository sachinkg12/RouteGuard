"""OpenAI-compatible LLM client (works with OpenAI, Together, vLLM, etc.).

The `openai` package is an optional dependency; importing this module raises a
clear message if it's missing. The mock client remains the default.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from .base import LLMClient, LLMResponse


class OpenAICompatibleClient(LLMClient):
    kind = "openai_compatible"

    def __init__(self, cfg) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised manually only
            raise ImportError(
                "OpenAICompatibleClient requires the 'openai' package. "
                "Install with: pip install ticket_routing[llm-openai]"
            ) from e

        api_key: Optional[str] = None
        if cfg.api_key_env:
            api_key = os.environ.get(cfg.api_key_env)
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.extra_body = getattr(cfg, "extra_body", None)
        self.cost_in = cfg.cost_per_1k_input_tokens
        self.cost_out = cfg.cost_per_1k_output_tokens
        self._client = OpenAI(
            api_key=api_key,
            base_url=cfg.base_url,
            timeout=cfg.request_timeout_s,
            max_retries=8,
        )

    def generate(self, prompt: str) -> LLMResponse:  # pragma: no cover - network
        t0 = time.perf_counter()
        extra = {"extra_body": self.extra_body} if self.extra_body else {}
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **extra,
        )
        latency = time.perf_counter() - t0
        text = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        cost = 0.0
        if usage is not None:
            cost = (
                (usage.prompt_tokens / 1000) * self.cost_in
                + (usage.completion_tokens / 1000) * self.cost_out
            )
        return LLMResponse(
            text=text,
            token_logprobs=None,
            raw=completion.model_dump() if hasattr(completion, "model_dump") else None,
            latency_seconds=latency,
            cost_estimate=cost,
            model=self.model,
            meta={"finish_reason": completion.choices[0].finish_reason},
        )
