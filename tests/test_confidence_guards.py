"""Regression tests for confidence and policy-composition safeguards."""
from __future__ import annotations

from ticket_routing.confidence.model_reported import ModelReportedConfidence
from ticket_routing.confidence.self_consistency import SelfConsistencyConfidence
from ticket_routing.evaluation.evaluator import Evaluator
from ticket_routing.models.base import PARSE_ERROR, PARSE_OK, PredictionBatch


def test_nonfinite_model_confidence_becomes_zero():
    batch = PredictionBatch(predicted_labels=["a"], confidence_scores=[float("nan")])
    assert ModelReportedConfidence().score(batch) == [0.0]


def test_self_consistency_reads_batch_not_mutable_predictor_state():
    batch = PredictionBatch(
        predicted_labels=["a", "b"],
        confidence_scores=[0.75, 1.0],
        parse_status=[PARSE_OK, PARSE_ERROR],
    )
    predictor_with_stale_state = type("P", (), {"_last_samples": [[{"label": "x"}]]})()
    assert SelfConsistencyConfidence(predictor_with_stale_state).score(batch) == [0.75, 0.0]


def test_agreement_policies_require_explicit_opt_in():
    auxiliary = [PredictionBatch(predicted_labels=["a"])]
    evaluator = Evaluator(
        cost_options=[5.0],
        default_wrong_cost=5.0,
        human_triage_cost=1.0,
        correct_auto_cost=0.0,
        thresholds=[0.5],
    )
    assert not any(p.name.startswith("agreement>=") for p in evaluator._build_policies(auxiliary))

    opted_in = Evaluator(
        cost_options=[5.0],
        default_wrong_cost=5.0,
        human_triage_cost=1.0,
        correct_auto_cost=0.0,
        thresholds=[0.5],
        include_agreement_policies=True,
    )
    assert any(p.name.startswith("agreement>=") for p in opted_in._build_policies(auxiliary))
