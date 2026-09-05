"""Threshold-based abstention: auto-route iff confidence >= threshold."""
from __future__ import annotations

from typing import List, Optional, Sequence

from .base import AUTO_ROUTE, DEFER_TO_HUMAN, AbstentionPolicy
from ..models.base import PARSE_OK, PredictionBatch


class ThresholdAbstentionPolicy(AbstentionPolicy):
    def __init__(self, threshold: float, defer_on_parse_failure: bool = True) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold
        self.defer_on_parse_failure = defer_on_parse_failure
        self.name = f"threshold@{threshold:.2f}"

    def decide(
        self,
        primary: PredictionBatch,
        confidences: Sequence[float],
        auxiliary: Optional[Sequence[PredictionBatch]] = None,
    ) -> List[str]:
        statuses = primary.parse_status or [PARSE_OK] * len(primary)
        decisions: List[str] = []
        for conf, status in zip(confidences, statuses):
            if self.defer_on_parse_failure and status != PARSE_OK:
                decisions.append(DEFER_TO_HUMAN)
                continue
            decisions.append(AUTO_ROUTE if conf >= self.threshold else DEFER_TO_HUMAN)
        return decisions
