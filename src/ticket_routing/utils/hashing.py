"""Stable hashing helpers for datasets and configs (reproducibility)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return sha256_text(canonical)


def sha256_iter_strings(items: Iterable[str]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()
