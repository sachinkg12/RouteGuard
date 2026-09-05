"""Agreement-based confidence across multiple predictors.

For each test example, confidence = (max votes for primary's predicted label) / N,
where N includes the primary and all auxiliary predictors.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .base import ConfidenceEstimator
from ..models.base import PredictionBatch


class AgreementConfidence(ConfidenceEstimator):
    name = "agreement"

    def score(
        self,
        primary: PredictionBatch,
        auxiliary: Optional[Sequence[PredictionBatch]] = None,
    ) -> List[float]:
        aux = list(auxiliary or [])
        if not aux:
            return [0.0] * len(primary)
        total_voters = 1 + len(aux)
        scores: List[float] = []
        for i, pred in enumerate(primary.predicted_labels):
            votes = 1  # primary's vote
            for other in aux:
                if i < len(other.predicted_labels) and other.predicted_labels[i] == pred:
                    votes += 1
            scores.append(votes / total_voters)
        return scores
