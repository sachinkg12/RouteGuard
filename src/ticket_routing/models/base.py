"""Predictor interface and PredictionBatch carrier.

Evaluator and abstention policies depend on this interface only; they never
inspect classical-vs-LLM internals.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


PARSE_OK = "OK"
PARSE_ERROR = "PARSE_ERROR"
INVALID_LABEL = "INVALID_LABEL"


@dataclass
class PredictionBatch:
    predicted_labels: List[str]
    confidence_scores: Optional[List[float]] = None
    raw_outputs: Optional[List[str]] = None
    parse_status: Optional[List[str]] = None  # PARSE_OK / PARSE_ERROR / INVALID_LABEL
    latency_seconds: Optional[List[float]] = None
    cost_estimate: Optional[List[float]] = None
    model_metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.predicted_labels)


class Predictor(abc.ABC):
    """All predictors must expose .fit() and .predict(); evaluator uses this only."""

    name: str = "base"

    @abc.abstractmethod
    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[str],
        label_names: Sequence[str],
    ) -> None: ...

    @abc.abstractmethod
    def predict(self, texts: Sequence[str]) -> PredictionBatch: ...
