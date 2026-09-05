"""Optional Platt / temperature-scaling-style calibration on the CALIBRATION split.

Kept intentionally simple: isotonic-regression-free, just temperature scaling on
softmax logits when available, or Platt on the predicted probability column.
This module is *opt-in*; the main experiments use raw model-reported confidence unless
the experiment config requests calibration.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattConfidenceCalibrator:
    """Univariate Platt-style scaling of confidence -> P(correct).

    Fit on calibration split using (confidence, correctness) pairs.
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self.fitted = False

    def fit(self, confidences: Sequence[float], correctness: Sequence[int]) -> None:
        X = np.asarray(confidences, dtype=float).reshape(-1, 1)
        y = np.asarray(correctness, dtype=int)
        if len(np.unique(y)) < 2:
            # Pathological calibration split (all-correct or all-wrong); skip.
            self.fitted = False
            return
        self._model = LogisticRegression(max_iter=1000)
        self._model.fit(X, y)
        self.fitted = True

    def transform(self, confidences: Sequence[float]) -> List[float]:
        if not self.fitted or self._model is None:
            return [float(c) for c in confidences]
        X = np.asarray(confidences, dtype=float).reshape(-1, 1)
        return self._model.predict_proba(X)[:, 1].tolist()
