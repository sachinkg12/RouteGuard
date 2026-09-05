"""Confidence-estimator interface.

Each strategy maps a PredictionBatch (plus optional auxiliary batches for
agreement) to a vector of confidence scores in [0, 1] of the same length.
"""
from __future__ import annotations

import abc
from typing import List, Optional, Sequence

from ..models.base import PredictionBatch


class ConfidenceEstimator(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def score(
        self,
        primary: PredictionBatch,
        auxiliary: Optional[Sequence[PredictionBatch]] = None,
    ) -> List[float]: ...
