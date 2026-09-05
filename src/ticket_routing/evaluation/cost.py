"""Configurable cost model and policy comparison metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..abstention.base import AUTO_ROUTE, DEFER_TO_HUMAN
from ..models.base import PredictionBatch


@dataclass(frozen=True)
class CostModel:
    correct_auto_route: float = 0.0
    human_triage: float = 1.0
    wrong_auto_route: float = 5.0


def per_ticket_costs(
    batch: PredictionBatch,
    true_labels: Sequence[str],
    decisions: Sequence[str],
    cost: CostModel,
) -> List[float]:
    out: List[float] = []
    for pred, truth, decision in zip(batch.predicted_labels, true_labels, decisions):
        if decision == DEFER_TO_HUMAN:
            out.append(cost.human_triage)
        elif decision == AUTO_ROUTE:
            out.append(cost.correct_auto_route if pred == truth else cost.wrong_auto_route)
        else:
            raise ValueError(f"Unknown decision: {decision}")
    return out


def expected_cost_per_ticket(per_ticket: Sequence[float]) -> float:
    if not per_ticket:
        return 0.0
    return float(np.mean(per_ticket))


def total_cost(per_ticket: Sequence[float]) -> float:
    return float(np.sum(per_ticket))


def relative_reduction(baseline: float, policy: float) -> float:
    if baseline == 0:
        return 0.0
    return float((baseline - policy) / baseline)


def cost_table_row(
    batch: PredictionBatch,
    true_labels: Sequence[str],
    decisions: Sequence[str],
    cost: CostModel,
    baseline_always_route_costs: Sequence[float],
    baseline_always_defer_cost_per_ticket: float,
) -> Dict:
    per_t = per_ticket_costs(batch, true_labels, decisions, cost)
    e_cost = expected_cost_per_ticket(per_t)
    base_ar = expected_cost_per_ticket(baseline_always_route_costs)
    return {
        "expected_cost_per_ticket": e_cost,
        "total_cost": total_cost(per_t),
        "cost_reduction_vs_always_route": relative_reduction(base_ar, e_cost),
        "cost_reduction_vs_always_defer": relative_reduction(
            baseline_always_defer_cost_per_ticket, e_cost
        ),
        "wrong_auto_route_cost": cost.wrong_auto_route,
        "human_triage_cost": cost.human_triage,
        "correct_auto_route_cost": cost.correct_auto_route,
        "per_ticket_costs": per_t,
    }
