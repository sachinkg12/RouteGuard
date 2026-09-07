"""Self-consistency confidence: vote share stored in a prediction batch.

``LLMPromptClassifier`` aggregates K samples into ``confidence_scores``.  The
estimator deliberately reads that immutable batch output instead of mutable
``predictor._last_samples`` state, which can otherwise refer to a later
calibration or test prediction call.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .base import ConfidenceEstimator
from ..models.base import PARSE_OK, PredictionBatch


class SelfConsistencyConfidence(ConfidenceEstimator):
    name = "self_consistency"

    def __init__(self, predictor=None) -> None:
        # Retained as an optional argument for backward-compatible configs.
        # Scoring is intentionally stateless.
        self.predictor = predictor

    def score(
        self,
        primary: PredictionBatch,
        auxiliary: Optional[Sequence[PredictionBatch]] = None,
    ) -> List[float]:
        if primary.confidence_scores is None:
            return [0.0] * len(primary)
        statuses = primary.parse_status or [PARSE_OK] * len(primary)
        return [
            min(1.0, max(0.0, float(confidence)))
            if status == PARSE_OK and confidence is not None
            else 0.0
            for confidence, status in zip(primary.confidence_scores, statuses)
        ]
