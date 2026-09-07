"""Tests for correctness discrimination and tie-aware risk--coverage metrics."""
from __future__ import annotations

import math

from ticket_routing.evaluation.selective import correctness_auroc, tie_group_aurc


def test_constant_confidence_has_chance_auroc_and_base_risk_aurc():
    correctness = [1, 0, 1, 0]
    confidence = [1.0, 1.0, 1.0, 1.0]
    assert correctness_auroc(confidence, correctness) == 0.5
    result = tie_group_aurc(confidence, correctness)
    assert math.isclose(result["aurc"], 0.5)
    assert result["n_unique_scores"] == 1
    assert len(result["curve"]) == 1


def test_tie_groups_are_admitted_together():
    result = tie_group_aurc([0.9, 0.9, 0.2], [1, 0, 0])
    assert result["curve"][0]["n_admitted"] == 2
    assert result["curve"][0]["risk"] == 0.5
    assert result["curve"][1]["n_admitted"] == 3
