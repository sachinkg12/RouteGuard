"""Deterministic LLM client for dry-run / tests / debug experiments.

Uses a tiny keyword-rule classifier to emit JSON in the expected schema.
No paid API. No network. Pure stdlib + numpy-free.
"""
from __future__ import annotations

import json
import time
from typing import List, Optional

from .base import LLMClient, LLMResponse


_KEYWORD_RULES: List[tuple[str, list[str]]] = [
    ("network", ["vpn", "wifi", "subnet", "dns", "packet", "switch", "internet"]),
    ("account_access", ["sso", "password", "mfa", "locked", "access", "permission", "group"]),
    ("hardware", ["battery", "monitor", "keyboard", "printer", "ssd", "laptop", "device"]),
    ("software_install", ["install", "plugin", "client", "push", "uninstall", "update", "creative suite"]),
    ("email", ["outlook", "mailbox", "calendar", "spam", "out of office", "email"]),
    ("security_incident", ["phishing", "malware", "edr", "leaked", "suspicious", "sign in", "attachment"]),
]


class DryRunMockClient(LLMClient):
    kind = "mock"

    def __init__(self, model: str = "mock-classifier", noise_rate: float = 0.0) -> None:
        self.model = model
        self.noise_rate = noise_rate
        self._call_count = 0

    def generate(self, prompt: str) -> LLMResponse:
        self._call_count += 1
        t0 = time.perf_counter()
        # Pull the allowed label set out of the prompt so the client is reusable
        # across datasets without knowing labels in advance.
        allowed = _extract_allowed_labels(prompt)
        ticket_text = _extract_ticket_text(prompt).lower()
        label, score = _rule_predict(ticket_text, allowed)
        # If allowed labels are something other than the keyword-rule names,
        # fall back to the first allowed label so JSON stays valid.
        if label not in allowed and allowed:
            label = allowed[0]
            score = 1.0 / max(1, len(allowed))
        payload = {"label": label, "confidence": round(score, 4)}
        text = json.dumps(payload)
        return LLMResponse(
            text=text,
            token_logprobs=None,
            raw=payload,
            latency_seconds=time.perf_counter() - t0,
            cost_estimate=0.0,
            model=self.model,
            meta={"call_index": self._call_count},
        )


def _extract_allowed_labels(prompt: str) -> List[str]:
    # Prompt format: "Choose exactly one label from this list:\n<labels>\n"
    marker = "Choose exactly one label from this list:"
    if marker not in prompt:
        return []
    after = prompt.split(marker, 1)[1]
    line = after.strip().splitlines()[0]
    return [tok.strip() for tok in line.split(",") if tok.strip()]


def _extract_ticket_text(prompt: str) -> str:
    if "Ticket:" not in prompt:
        return prompt
    after = prompt.rsplit("Ticket:", 1)[1]
    return after.split("Return JSON only", 1)[0].strip()


def _rule_predict(text: str, allowed: List[str]) -> tuple[str, float]:
    allowed_set = set(allowed)
    scores: dict[str, int] = {}
    for label, kws in _KEYWORD_RULES:
        if allowed_set and label not in allowed_set:
            continue
        hits = sum(1 for kw in kws if kw in text)
        if hits:
            scores[label] = hits
    if not scores:
        # Even split: pretend low confidence on first allowed label.
        if allowed:
            return allowed[0], max(0.34, 1.0 / max(1, len(allowed)))
        return "unknown", 0.0
    total = sum(scores.values())
    best = max(scores.items(), key=lambda kv: kv[1])
    confidence = 0.5 + 0.5 * (best[1] / total)
    return best[0], min(0.99, confidence)
