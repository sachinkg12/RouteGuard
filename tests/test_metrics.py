"""Classification metrics + abstention metrics behave correctly."""
from __future__ import annotations

import math

from ticket_routing.evaluation.metrics import (
    compute_abstention_metrics,
    compute_classification_metrics,
)
from ticket_routing.models.base import PARSE_OK, PredictionBatch


def _batch(preds, statuses=None):
    return PredictionBatch(
        predicted_labels=preds,
        confidence_scores=[1.0] * len(preds),
        parse_status=statuses or [PARSE_OK] * len(preds),
    )


def test_classification_metrics_perfect_predictions():
    batch = _batch(["a", "b", "a"])
    m = compute_classification_metrics(batch, ["a", "b", "a"], ["a", "b"])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["parse_error_rate"] == 0.0


def test_classification_metrics_mixed():
    batch = _batch(["a", "a", "b"])
    m = compute_classification_metrics(batch, ["a", "b", "b"], ["a", "b"])
    assert math.isclose(m["accuracy"], 2 / 3)


def test_abstention_metrics_all_routed():
    batch = _batch(["a", "b", "a"])
    decisions = ["AUTO_ROUTE"] * 3
    m = compute_abstention_metrics(batch, ["a", "b", "b"], decisions)
    assert m["coverage"] == 1.0
    assert m["deferred_rate"] == 0.0
    # 2/3 correct, 1/3 wrong -> wrong_auto_route_rate = 1/3
    assert math.isclose(m["wrong_auto_route_rate"], 1 / 3)
    assert math.isclose(m["accuracy_on_auto_routed"], 2 / 3)


def test_abstention_metrics_all_deferred():
    batch = _batch(["a", "b"])
    decisions = ["DEFER_TO_HUMAN", "DEFER_TO_HUMAN"]
    m = compute_abstention_metrics(batch, ["a", "b"], decisions)
    assert m["coverage"] == 0.0
    assert m["deferred_rate"] == 1.0
    assert m["wrong_auto_route_rate"] == 0.0
    assert math.isnan(m["accuracy_on_auto_routed"])
