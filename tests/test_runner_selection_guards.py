"""The runner evaluates exactly the operating point frozen on calibration."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from ticket_routing.reporting.tables import cost_table


_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_classification.py"
_SPEC = importlib.util.spec_from_file_location("routeguard_run_classification", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)
_attach_bootstrap_cis = _RUNNER._attach_bootstrap_cis


def _cost_row(policy, expected, per_ticket):
    return {
        "policy": policy,
        "wrong_auto_route_cost": 5.0,
        "expected_cost_per_ticket": expected,
        "total_cost": sum(per_ticket),
        "cost_reduction_vs_always_route": 0.0,
        "cost_reduction_vs_always_defer": 0.0,
        "per_ticket_costs": per_ticket,
    }


def test_bootstrap_uses_calibration_selected_policy_not_test_minimum():
    # threshold@0.9 has lower test cost, but calibration selected threshold@0.5.
    # The test helper must not optimize or switch policies.
    result = SimpleNamespace(
        cost=[
            _cost_row("always_route", 2.5, [0.0, 5.0, 0.0, 5.0]),
            _cost_row("threshold@0.5", 1.0, [0.0, 1.0, 1.0, 2.0]),
            _cost_row("threshold@0.9", 0.5, [0.0, 1.0, 0.0, 1.0]),
        ],
        metadata={
            "policy_selection": {
                "policy": "threshold@0.5",
                "selection_split": "calibration",
            }
        },
        predictor_name="demo",
        setting="test",
        confidence_method="model_reported",
    )
    cfg = SimpleNamespace(
        cost=SimpleNamespace(
            correct_auto_route=0.0,
            human_triage=1.0,
            default_wrong_auto_route=5.0,
        ),
        bootstrap=SimpleNamespace(samples=200, seed=7),
    )

    _attach_bootstrap_cis([result], cfg)

    relative = result.metadata["bootstrap"][
        "selected_threshold_vs_always_route_relative_reduction"
    ]
    assert relative["selected_policy"] == "threshold@0.5"
    assert relative["selection_split"] == "calibration"


def test_cost_table_places_interval_only_on_selected_row():
    result = SimpleNamespace(
        cost=[
            _cost_row("always_route", 2.5, [0.0, 5.0, 0.0, 5.0]),
            _cost_row("threshold@0.5", 1.0, [0.0, 1.0, 1.0, 2.0]),
            _cost_row("threshold@0.9", 0.5, [0.0, 1.0, 0.0, 1.0]),
        ],
        metadata={
            "bootstrap": {
                "selected_threshold_vs_always_route_cost_difference": {
                    "selected_policy": "threshold@0.5",
                    "wrong_route_cost": 5.0,
                    "selection_split": "calibration",
                    "ci_low": -2.0,
                    "ci_high": -1.0,
                },
                "selected_threshold_vs_always_route_relative_reduction": {
                    "ci_low": 0.4,
                    "ci_high": 0.8,
                },
            }
        },
        predictor_name="demo",
        setting="test",
        confidence_method="model_reported",
    )

    table = cost_table([result])
    selected = table[table["policy"] == "threshold@0.5"].iloc[0]
    unselected = table[table["policy"] == "threshold@0.9"].iloc[0]
    assert selected["relative_cost_reduction_ci_low"] == 0.4
    assert selected["threshold_selection_split"] == "calibration"
    assert unselected["relative_cost_reduction_ci_low"] != unselected[
        "relative_cost_reduction_ci_low"
    ]  # NaN


def test_zero_cost_baseline_records_undefined_relative_reduction():
    result = SimpleNamespace(
        cost=[
            _cost_row("always_route", 0.0, [0.0, 0.0]),
            _cost_row("threshold@0.5", 0.0, [0.0, 0.0]),
        ],
        metadata={
            "policy_selection": {
                "policy": "threshold@0.5",
                "selection_split": "calibration",
            }
        },
    )
    cfg = SimpleNamespace(
        cost=SimpleNamespace(
            correct_auto_route=0.0,
            human_triage=1.0,
            default_wrong_auto_route=5.0,
        ),
        bootstrap=SimpleNamespace(samples=20, seed=7),
    )

    _attach_bootstrap_cis([result], cfg)

    relative = result.metadata["bootstrap"][
        "selected_threshold_vs_always_route_relative_reduction"
    ]
    assert relative["point_estimate"] is None
    assert "zero-cost baseline" in relative["undefined_reason"]
