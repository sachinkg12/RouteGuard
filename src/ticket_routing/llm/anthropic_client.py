"""Anthropic Messages API client.

The Anthropic API is not OpenAI-compatible, so we need a dedicated client.
The `anthropic` package is optional; importing this module raises a clear
message if it's missing.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from .base import LLMClient, LLMResponse


class AnthropicClient(LLMClient):
    kind = "anthropic"

    def __init__(self, cfg) -> None:
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised manually only
            raise ImportError(
                "AnthropicClient requires the 'anthropic' package. "
                "Install with: pip install anthropic"
            ) from e

        api_key: Optional[str] = None
        if cfg.api_key_env:
            api_key = os.environ.get(cfg.api_key_env)
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.cost_in = cfg.cost_per_1k_input_tokens
        self.cost_out = cfg.cost_per_1k_output_tokens
        # max_retries=8 with the SDK's default exponential backoff handles
        # transient 429/5xx/529 overloads up to several minutes of outage.
        self._client = Anthropic(
            api_key=api_key,
            timeout=cfg.request_timeout_s,
            max_retries=8,
        )

    def generate(self, prompt: str) -> LLMResponse:  # pragma: no cover - network
        t0 = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - t0
        # Concatenate all text blocks (usually one).
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(message, "usage", None)
        cost = 0.0
        if usage is not None:
            cost = (
                (usage.input_tokens / 1000) * self.cost_in
                + (usage.output_tokens / 1000) * self.cost_out
            )
        return LLMResponse(
            text=text,
            token_logprobs=None,
            raw=message.model_dump() if hasattr(message, "model_dump") else None,
            latency_seconds=latency,
            cost_estimate=cost,
            model=self.model,
            meta={"stop_reason": getattr(message, "stop_reason", None)},
        )
