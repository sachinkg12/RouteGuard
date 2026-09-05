"""Expected Calibration Error (ECE) and reliability data."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[int],
    n_bins: int = 10,
) -> Dict:
    conf = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    corr = np.asarray(correctness, dtype=float)
    if len(conf) != len(corr):
        raise ValueError("confidences and correctness must align")
    if len(conf) == 0:
        return {"ece": 0.0, "n_bins": n_bins, "bins": []}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Right edge inclusive only on the final bin.
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            bins.append(
                {"bin": [float(lo), float(hi)], "count": 0, "avg_conf": None, "accuracy": None}
            )
            continue
        avg_conf = float(conf[mask].mean())
        acc = float(corr[mask].mean())
        weight = mask.sum() / n
        ece += weight * abs(avg_conf - acc)
        bins.append(
            {"bin": [float(lo), float(hi)], "count": int(mask.sum()), "avg_conf": avg_conf, "accuracy": acc}
        )
    return {"ece": float(ece), "n_bins": n_bins, "bins": bins}


def correctness_vector(
    predicted_labels: Sequence[str],
    true_labels: Sequence[str],
) -> List[int]:
    return [int(p == t) for p, t in zip(predicted_labels, true_labels)]
