"""Model-reported confidence: classical probability or LLM JSON `confidence` field."""
from __future__ import annotations

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
        # (LLM parser coerces; classical uses softmax), but enforce the invariant
        # here so a future predictor emitting an out-of-range value cannot distort
        # mean-confidence or abstention downstream.
        return [
            min(1.0, max(0.0, float(c))) if c is not None else 0.0
            for c in primary.confidence_scores
        ]
