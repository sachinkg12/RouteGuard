"""Discrimination metrics for confidence-based selective prediction."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def correctness_auroc(confidences: Sequence[float], correctness: Sequence[int]) -> float:
    """AUROC for ranking correct predictions above incorrect predictions.

    Ties receive the standard half credit used by ``roc_auc_score``. A constant
    score therefore has AUROC 0.5 when both correctness classes are present.
    """

    scores, outcomes = _validated_arrays(confidences, correctness)
    if np.unique(outcomes).size < 2:
        return float("nan")
    return float(roc_auc_score(outcomes, scores))


def tie_group_aurc(confidences: Sequence[float], correctness: Sequence[int]) -> Dict:
    """Area under a tie-aware risk--coverage curve.

    Tickets are admitted in descending confidence order, but an entire tied
    confidence group is admitted at once. Risk after each admission is the error
    rate among all admitted tickets. The area is the right-continuous step integral
    ``sum(delta_coverage * risk_after_group)``. This avoids arbitrary ordering
    within the many tied scores produced by prompted LLMs.
    """

    scores, outcomes = _validated_arrays(confidences, correctness)
    if scores.size == 0:
        return {"aurc": float("nan"), "tie_handling": "whole_group", "curve": []}

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_errors = 1.0 - outcomes[order]
    n = scores.size
    admitted = 0
    errors = 0.0
    area = 0.0
    curve = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_n = end - start
        admitted += group_n
        errors += float(sorted_errors[start:end].sum())
        coverage = admitted / n
        risk = errors / admitted
        area += (group_n / n) * risk
        curve.append(
            {
                "threshold": float(sorted_scores[start]),
                "coverage": float(coverage),
                "risk": float(risk),
                "n_admitted": int(admitted),
                "tie_group_size": int(group_n),
            }
        )
        start = end

    return {
        "aurc": float(area),
        "tie_handling": "whole_group_right_continuous_step_integral",
        "n_unique_scores": int(np.unique(scores).size),
        "curve": curve,
    }


def _validated_arrays(
    confidences: Sequence[float], correctness: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(confidences, dtype=float)
    outcomes = np.asarray(correctness, dtype=int)
    if scores.ndim != 1 or outcomes.ndim != 1 or scores.size != outcomes.size:
        raise ValueError("confidences and correctness must be equal-length vectors")
    if not np.all(np.isfinite(scores)):
        raise ValueError("confidences must be finite")
    if not np.all(np.isin(outcomes, [0, 1])):
        raise ValueError("correctness values must be 0 or 1")
    return scores, outcomes
