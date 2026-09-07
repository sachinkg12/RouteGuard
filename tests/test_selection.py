"""Operating points are selected on calibration and frozen for test."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ticket_routing.evaluation.cost import CostModel
from ticket_routing.evaluation.metrics import bootstrap_relative_reduction
from ticket_routing.evaluation.selection import select_threshold_on_calibration
from ticket_routing.models.base import PARSE_OK, PredictionBatch


def _batch(predictions):
    return PredictionBatch(
        predicted_labels=list(predictions),
        parse_status=[PARSE_OK] * len(predictions),
    )


def test_threshold_selection_records_calibration_provenance():
    batch = _batch(["a", "x", "b", "x"])
    selection = select_threshold_on_calibration(
        batch=batch,
        true_labels=["a", "a", "b", "b"],
        confidences=[0.9, 0.8, 0.7, 0.6],
        thresholds=[0.5, 0.75, 0.85],
        cost=CostModel(0.0, 1.0, 5.0),
    )

    assert selection.selection_split == "calibration"
    assert selection.policy == "threshold@0.85"
    assert selection.threshold == 0.85
    assert selection.threshold_grid == (0.5, 0.75, 0.85)
    assert selection.n_selection_examples == 4
    assert selection.cost_parameters["wrong_auto_route"] == 5.0


def test_threshold_selection_rejects_test_provenance():
    with pytest.raises(ValueError, match="calibration split"):
        select_threshold_on_calibration(
            batch=_batch(["a"]),
            true_labels=["a"],
            confidences=[0.9],
            thresholds=[0.5],
            cost=CostModel(),
            selection_split="test",
        )


def test_relative_reduction_bootstraps_the_ratio_estimand():
    result = bootstrap_relative_reduction(
        policy_costs=[0.0, 1.0, 0.0, 1.0],
        baseline_costs=[0.0, 5.0, 0.0, 5.0],
        n_samples=200,
        seed=7,
    )

    assert result["estimand"] == "relative_cost_reduction"
    assert math.isclose(result["point_estimate"], 0.8)
    assert 0.0 <= result["ci_low"] <= result["ci_high"] <= 1.0


def test_relative_reduction_accepts_numpy_vectors():
    result = bootstrap_relative_reduction(
        policy_costs=np.array([0.0, 1.0]),
        baseline_costs=np.array([0.0, 5.0]),
        n_samples=20,
        seed=7,
    )
    assert math.isclose(result["point_estimate"], 0.8)
