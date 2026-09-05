"""Post-hoc isotonic calibration of model-reported confidences.

This is the recommended baseline for re-calibrating LLM verbalized
``confidence`` fields, which are bounded in [0, 1] but are not logits ---
so parametric Platt/temperature scaling does not apply cleanly.
Isotonic regression makes no parametric assumption and learns a monotone
non-decreasing mapping ``raw confidence -> P(correct)`` from a held-out
calibration split.

Usage in the runner:
    estimator = IsotonicConfidence()
    estimator.fit_from_pairs(raw_confs, correctness)
    test_confs = estimator.score(primary_batch)
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .base import ConfidenceEstimator
from ..models.base import PredictionBatch


class IsotonicConfidence(ConfidenceEstimator):
    """Isotonic-regression recalibration of model-reported confidence.

    ``fit_from_pairs`` is called once per predictor on the calibration
    split before evaluation. ``score`` then applies the fitted monotone
    mapping to the primary batch's raw confidences.
    """

    name = "isotonic"

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None
        self._fitted: bool = False
        self._n_calib: int = 0

    def fit_from_pairs(
        self,
        confidences: Sequence[float],
        correctness: Sequence[int],
    ) -> None:
        x = np.asarray(
            [float(c) if c is not None else 0.0 for c in confidences],
            dtype=float,
        )
        y = np.asarray(correctness, dtype=float)
        if x.size == 0 or len(np.unique(y)) < 2:
            self._fitted = False
            self._n_calib = int(x.size)
            return
        self._model = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True
        )
        self._model.fit(x, y)
        self._fitted = True
        self._n_calib = int(x.size)

    def score(
        self,
        primary: PredictionBatch,
        auxiliary: Optional[Sequence[PredictionBatch]] = None,
    ) -> List[float]:
        if primary.confidence_scores is None:
            return [0.0] * len(primary)
        raw = [float(c) if c is not None else 0.0 for c in primary.confidence_scores]
        if not self._fitted or self._model is None:
            # Fall through to raw if not fitted (e.g. degenerate calibration split).
            return raw
        return [float(v) for v in self._model.predict(np.asarray(raw, dtype=float))]

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def n_calibration_pairs(self) -> int:
        return self._n_calib
