"""Model-reported confidence: classical probability or LLM JSON `confidence` field."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .base import ConfidenceEstimator
from ..models.base import PredictionBatch


class ModelReportedConfidence(ConfidenceEstimator):
    name = "model_reported"

    def score(
        self,
        primary: PredictionBatch,
        auxiliary: Optional[Sequence[PredictionBatch]] = None,
    ) -> List[float]:
        if primary.confidence_scores is None:
            return [0.0] * len(primary)
        # Defensive clip: model-reported confidences should already be in [0, 1]
        # (the LLM parser coerces it; non-LLM predictors expose bounded class-
        # probability estimates), but enforce the invariant
        # here so a future predictor emitting an out-of-range value cannot distort
        # mean-confidence or abstention downstream.
        scores: List[float] = []
        for confidence in primary.confidence_scores:
            value = float(confidence) if confidence is not None else 0.0
            scores.append(min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0)
        return scores
