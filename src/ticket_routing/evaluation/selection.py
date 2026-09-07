"""Calibration-only operating-point selection.

The test split must never be passed to this module.  Selection returns enough
metadata for result bundles to prove where and how the policy was chosen.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

from ..abstention.threshold_policy import ThresholdAbstentionPolicy
from ..models.base import PredictionBatch
from .cost import CostModel, expected_cost_per_ticket, per_ticket_costs


@dataclass(frozen=True)
class ThresholdSelection:
    """Frozen threshold selected on a named non-test split."""

    threshold: float
    policy: str
    expected_cost_per_ticket: float
    candidate_costs: Dict[str, float]
    threshold_grid: tuple[float, ...]
    selection_split: str
    n_selection_examples: int
    cost_parameters: Dict[str, float]

    def to_metadata(self) -> dict:
        return asdict(self)


def select_threshold_on_calibration(
    *,
    batch: PredictionBatch,
    true_labels: Sequence[str],
    confidences: Sequence[float],
    thresholds: Sequence[float],
    cost: CostModel,
    selection_split: str = "calibration",
) -> ThresholdSelection:
    """Select the minimum-cost threshold on calibration data only.

    ``selection_split`` is intentionally validated so a caller cannot quietly
    relabel test-set optimization as deployment-style threshold selection.
    """

    if selection_split != "calibration":
        raise ValueError("reported threshold selection must use the calibration split")
    if len(batch) != len(true_labels) or len(batch) != len(confidences):
        raise ValueError("predictions, labels, and confidences must have equal length")
    if not thresholds:
        raise ValueError("at least one threshold is required")

    grid = tuple(float(t) for t in thresholds)
    candidate_costs: Dict[str, float] = {}
    for threshold in grid:
        policy = ThresholdAbstentionPolicy(threshold=threshold)
        decisions = policy.decide(batch, confidences)
        candidate_costs[policy.name] = expected_cost_per_ticket(
            per_ticket_costs(batch, true_labels, decisions, cost)
        )

    selected_policy = min(candidate_costs, key=candidate_costs.get)
    selected_threshold = float(selected_policy.split("@", 1)[1])
    return ThresholdSelection(
        threshold=selected_threshold,
        policy=selected_policy,
        expected_cost_per_ticket=candidate_costs[selected_policy],
        candidate_costs=candidate_costs,
        threshold_grid=grid,
        selection_split=selection_split,
        n_selection_examples=len(batch),
        cost_parameters={
            "correct_auto_route": float(cost.correct_auto_route),
            "human_triage": float(cost.human_triage),
            "wrong_auto_route": float(cost.wrong_auto_route),
        },
    )
