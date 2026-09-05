"""Strict JSON parser for LLM classification output.

Rules:
1. Accept only valid JSON.
2. Accept only labels exactly matching the allowed label set.
3. Out-of-set labels are tagged INVALID_LABEL (not silently mapped).
4. JSON / shape failures are tagged PARSE_ERROR.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

from ..models.base import INVALID_LABEL, PARSE_ERROR, PARSE_OK


@dataclass
class ParsedLLMOutput:
    label: str
    confidence: Optional[float]
    status: str  # PARSE_OK / PARSE_ERROR / INVALID_LABEL


def parse_label_json(text: str, allowed_labels: Sequence[str]) -> ParsedLLMOutput:
    if not text or not isinstance(text, str):
        return ParsedLLMOutput("__PARSE_ERROR__", None, PARSE_ERROR)

    payload = _try_parse_json(text)
    if payload is None or not isinstance(payload, dict):
        return ParsedLLMOutput("__PARSE_ERROR__", None, PARSE_ERROR)

    label = payload.get("label")
    if not isinstance(label, str):
        return ParsedLLMOutput("__PARSE_ERROR__", None, PARSE_ERROR)

    label = label.strip()
    if label not in set(allowed_labels):
        return ParsedLLMOutput(label, _coerce_confidence(payload.get("confidence")), INVALID_LABEL)

    return ParsedLLMOutput(label, _coerce_confidence(payload.get("confidence")), PARSE_OK)


def _try_parse_json(text: str):
    text = text.strip()
    # Direct parse first; otherwise extract the first *balanced* JSON object and
    # try each in order. A balanced-brace scan (not a greedy `{.*}` regex) is
    # robust to preambles/suffixes that themselves contain braces, e.g. a
    # reasoning "<think>the set is {a,b}</think>" block preceding the real JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for block in _iter_json_objects(text):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    return None


def _iter_json_objects(text: str) -> Iterator[str]:
    """Yield each top-level balanced ``{...}`` substring, respecting JSON strings."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    break
        start = text.find("{", start + 1)


def _coerce_confidence(value) -> Optional[float]:
    if value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf
