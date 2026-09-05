"""Cost model returns expected per-ticket and aggregated costs."""
from __future__ import annotations

import math

from ticket_routing.evaluation.cost import (
    CostModel,
    expected_cost_per_ticket,
    per_ticket_costs,
    relative_reduction,
)
from ticket_routing.models.base import PredictionBatch


def _batch(preds):
    return PredictionBatch(predicted_labels=preds)


def test_per_ticket_cost_breakdown():
    cm = CostModel(correct_auto_route=0.0, human_triage=1.0, wrong_auto_route=5.0)
    batch = _batch(["a", "a", "b", "c"])
    decisions = ["AUTO_ROUTE", "AUTO_ROUTE", "DEFER_TO_HUMAN", "AUTO_ROUTE"]
    truth = ["a", "b", "b", "x"]  # i=0 correct, i=1 wrong, i=2 deferred, i=3 wrong
    costs = per_ticket_costs(batch, truth, decisions, cm)
    assert costs == [0.0, 5.0, 1.0, 5.0]
    assert math.isclose(expected_cost_per_ticket(costs), (0 + 5 + 1 + 5) / 4)


def test_relative_reduction():
    assert relative_reduction(1.0, 0.5) == 0.5
    assert relative_reduction(0.0, 0.5) == 0.0  # safe-guards against divide-by-zero
    assert relative_reduction(2.0, 2.0) == 0.0


def test_cost_model_uses_wrong_route_cost():
    cm = CostModel(correct_auto_route=0.0, human_triage=1.0, wrong_auto_route=10.0)
    batch = _batch(["a"])
    costs = per_ticket_costs(batch, ["b"], ["AUTO_ROUTE"], cm)
    assert costs == [10.0]
