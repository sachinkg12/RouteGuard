"""Agreement-based abstention: auto-route iff at least N predictors agree."""
from __future__ import annotations

from typing import List, Optional, Sequence

from .base import AUTO_ROUTE, DEFER_TO_HUMAN, AbstentionPolicy
from ..models.base import PARSE_OK, PredictionBatch


class AgreementAbstentionPolicy(AbstentionPolicy):
    def __init__(self, min_agree: int) -> None:
        if min_agree < 1:
            raise ValueError("min_agree must be >= 1")
        self.min_agree = min_agree
        self.name = f"agreement>={min_agree}"

    def decide(
        self,
        primary: PredictionBatch,
        confidences: Sequence[float],
        auxiliary: Optional[Sequence[PredictionBatch]] = None,
    ) -> List[str]:
        aux = list(auxiliary or [])
        statuses = primary.parse_status or [PARSE_OK] * len(primary)
        decisions: List[str] = []
        for i, (pred, status) in enumerate(zip(primary.predicted_labels, statuses)):
            if status != PARSE_OK:
                decisions.append(DEFER_TO_HUMAN)
                continue
            agree = 1  # primary itself
            for other in aux:
                if i < len(other.predicted_labels) and other.predicted_labels[i] == pred:
                    agree += 1
            decisions.append(AUTO_ROUTE if agree >= self.min_agree else DEFER_TO_HUMAN)
        return decisions
