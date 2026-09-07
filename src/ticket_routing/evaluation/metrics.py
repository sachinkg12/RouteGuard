"""Core classification metrics on raw predictions (no abstention)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ..models.base import PARSE_ERROR, PARSE_OK, INVALID_LABEL, PredictionBatch


def compute_classification_metrics(
    batch: PredictionBatch,
    true_labels: Sequence[str],
    label_names: Sequence[str],
) -> Dict:
    pred = batch.predicted_labels
    statuses = batch.parse_status or [PARSE_OK] * len(pred)

    # For accuracy/F1, treat parse failures as wrong predictions (won't match any label).
    # This is the conservative interpretation used here.
    n = len(true_labels)
    parse_error_rate = sum(1 for s in statuses if s == PARSE_ERROR) / max(1, n)
    invalid_label_rate = sum(1 for s in statuses if s == INVALID_LABEL) / max(1, n)

    accuracy = accuracy_score(true_labels, pred)
    macro_f1 = f1_score(
        true_labels, pred, labels=list(label_names), average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        true_labels, pred, labels=list(label_names), average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        true_labels, pred, labels=list(label_names), zero_division=0
    )
    cm = confusion_matrix(true_labels, pred, labels=list(label_names)).tolist()

    return {
        "n": n,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            "labels": list(label_names),
            "precision": per_class[0].tolist(),
            "recall": per_class[1].tolist(),
            "f1": per_class[2].tolist(),
            "support": per_class[3].tolist(),
        },
        "confusion_matrix": cm,
        "parse_error_rate": float(parse_error_rate),
        "invalid_label_rate": float(invalid_label_rate),
    }


def compute_abstention_metrics(
    batch: PredictionBatch,
    true_labels: Sequence[str],
    decisions: Sequence[str],
) -> Dict:
    n = len(true_labels)
    auto_mask = np.array([d == "AUTO_ROUTE" for d in decisions])
    deferred_mask = ~auto_mask
    coverage = float(auto_mask.mean()) if n else 0.0
    deferred_rate = float(deferred_mask.mean()) if n else 0.0

    if auto_mask.sum() == 0:
        accuracy_on_auto = float("nan")
        wrong_auto_rate = 0.0
    else:
        pred_arr = np.array(batch.predicted_labels)
        true_arr = np.array(true_labels)
        correct_auto = (pred_arr[auto_mask] == true_arr[auto_mask]).sum()
        total_auto = int(auto_mask.sum())
        accuracy_on_auto = float(correct_auto / total_auto)
        wrong_auto_rate = float((total_auto - correct_auto) / n)

    return {
        "n": n,
        "coverage": coverage,
        "deferred_rate": deferred_rate,
        "n_auto_routed": int(auto_mask.sum()),
        "n_deferred": int(deferred_mask.sum()),
        "accuracy_on_auto_routed": accuracy_on_auto,
        "wrong_auto_route_rate": wrong_auto_rate,
    }


def bootstrap_difference(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_samples: int = 1000,
    seed: int = 1337,
) -> Dict:
    """Bootstrap 95% CI for mean(values_a) - mean(values_b)."""
    if len(values_a) != len(values_b):
        raise ValueError("paired bootstrap requires equal-length arrays")
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diffs = np.empty(n_samples)
    n = len(a)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return {
        "point_estimate": float(a.mean() - b.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "n_bootstrap": n_samples,
        "seed": seed,
    }


def bootstrap_relative_reduction(
    policy_costs: Sequence[float],
    baseline_costs: Sequence[float],
    n_samples: int = 1000,
    seed: int = 1337,
) -> Dict:
    """Paired-bootstrap CI for ``1 - mean(policy) / mean(baseline)``.

    The ratio is recomputed inside every resample.  This matches a reported
    relative-cost-reduction estimand; it is not an absolute-difference interval
    divided by the full-sample baseline after bootstrapping.
    """

    if len(policy_costs) != len(baseline_costs):
        raise ValueError("paired bootstrap requires equal-length arrays")
    if len(policy_costs) == 0:
        raise ValueError("paired bootstrap requires at least one observation")

    rng = np.random.default_rng(seed)
    policy = np.asarray(policy_costs, dtype=float)
    baseline = np.asarray(baseline_costs, dtype=float)
    baseline_mean = float(baseline.mean())
    if baseline_mean == 0.0:
        raise ValueError("relative reduction is undefined for a zero-cost baseline")

    reductions = np.empty(n_samples)
    n = len(policy)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        sampled_baseline = float(baseline[idx].mean())
        reductions[i] = (
            np.nan
            if sampled_baseline == 0.0
            else 1.0 - float(policy[idx].mean()) / sampled_baseline
        )
    finite = reductions[np.isfinite(reductions)]
    if finite.size == 0:
        raise ValueError("all bootstrap resamples had a zero-cost baseline")

    return {
        "estimand": "relative_cost_reduction",
        "point_estimate": float(1.0 - policy.mean() / baseline_mean),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "n_bootstrap": int(n_samples),
        "n_finite_bootstrap": int(finite.size),
        "seed": int(seed),
    }
