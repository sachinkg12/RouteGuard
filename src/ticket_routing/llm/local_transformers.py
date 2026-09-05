"""Local HuggingFace transformers LLM client (Phi-3, Llama 3.2, Gemma 2, ...).

Optional. Importing requires `transformers` + `torch`. Default config still uses
the mock client.
"""
from __future__ import annotations

import time

from .base import LLMClient, LLMResponse


class LocalTransformersClient(LLMClient):
    kind = "local_transformers"

    def __init__(self, cfg) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised manually only
            raise ImportError(
                "LocalTransformersClient requires 'transformers' + 'torch'. "
                "Install with: pip install ticket_routing[llm-local]"
            ) from e

        self.model_name = cfg.model
        self.temperature = max(cfg.temperature, 1e-3)
        self.max_tokens = cfg.max_tokens
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model = self.model_name

    def generate(self, prompt: str) -> LLMResponse:  # pragma: no cover - heavy
        t0 = time.perf_counter()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0.01,
                temperature=self.temperature,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return LLMResponse(
            text=text,
            token_logprobs=None,
            raw=None,
            latency_seconds=time.perf_counter() - t0,
            cost_estimate=0.0,
            model=self.model_name,
        )
