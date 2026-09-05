"""ECE returns expected values on toy distributions."""
from __future__ import annotations

import math

from ticket_routing.evaluation.calibration import expected_calibration_error


def test_ece_perfect_calibration_is_zero():
    # All predictions confidence=1.0 and all correct -> ECE = 0.
    out = expected_calibration_error([1.0] * 10, [1] * 10, n_bins=10)
    assert math.isclose(out["ece"], 0.0)


def test_ece_high_confidence_all_wrong_is_one():
    # Confidence 1.0 but accuracy 0.0 -> |1 - 0| = 1.
    out = expected_calibration_error([1.0] * 8, [0] * 8, n_bins=10)
    assert math.isclose(out["ece"], 1.0)


def test_ece_empty_inputs():
    out = expected_calibration_error([], [], n_bins=10)
    assert out["ece"] == 0.0


def test_ece_two_bins_split():
    # Half at conf=0.1 (all wrong), half at conf=0.9 (all correct) -> ECE = 0.1.
    confs = [0.1] * 50 + [0.9] * 50
    corrs = [0] * 50 + [1] * 50
    out = expected_calibration_error(confs, corrs, n_bins=10)
    # bin near 0.1 contributes |0.1 - 0| * 0.5; bin near 0.9 contributes |0.9-1|*0.5.
    assert math.isclose(out["ece"], 0.1, abs_tol=1e-9)
