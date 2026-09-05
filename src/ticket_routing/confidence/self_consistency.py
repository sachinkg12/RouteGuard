"""Self-consistency confidence: vote share across K LLM samples.

Relies on LLMPromptClassifier having recorded per-sample outputs in
`_last_samples`. If unavailable, falls back to model-reported confidence.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .base import ConfidenceEstimator
from ..models.base import PARSE_OK, PredictionBatch


class SelfConsistencyConfidence(ConfidenceEstimator):
    name = "self_consistency"

    def __init__(self, predictor) -> None:
        self.predictor = predictor

    def score(
        self,
        primary: PredictionBatch,
        auxiliary: Optional[Sequence[PredictionBatch]] = None,
    ) -> List[float]:
        last_samples = getattr(self.predictor, "_last_samples", None)
        if not last_samples or len(last_samples) != len(primary):
            # Fallback: model-reported.
            if primary.confidence_scores is None:
                return [0.0] * len(primary)
            return [float(c) if c is not None else 0.0 for c in primary.confidence_scores]

        scores: List[float] = []
        for samples, pred_label, pred_status in zip(
            last_samples, primary.predicted_labels, primary.parse_status or []
        ):
            if pred_status != PARSE_OK or not samples:
                scores.append(0.0)
                continue
            valid_matches = sum(
                1 for s in samples if s.get("status") == PARSE_OK and s.get("label") == pred_label
            )
            scores.append(valid_matches / len(samples))
        return scores
