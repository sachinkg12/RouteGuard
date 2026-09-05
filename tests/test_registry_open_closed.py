"""Adding a new predictor through the registry does not require evaluator edits."""
from __future__ import annotations

from typing import Sequence

from ticket_routing.evaluation.evaluator import Evaluator
from ticket_routing.confidence.model_reported import ModelReportedConfidence
from ticket_routing.models.base import PARSE_OK, PredictionBatch, Predictor
from ticket_routing.models.registry import build, register, registered_names


@register("constant_predictor_for_test")
class ConstantPredictor(Predictor):
    name = "constant_predictor_for_test"

    def __init__(self, label: str = "always_a") -> None:
        self.label = label

    def fit(self, train_texts, train_labels, label_names):
        self._labels = list(label_names)

    def predict(self, texts: Sequence[str]) -> PredictionBatch:
        return PredictionBatch(
            predicted_labels=[self.label] * len(texts),
            confidence_scores=[0.42] * len(texts),
            parse_status=[PARSE_OK] * len(texts),
            cost_estimate=[0.0] * len(texts),
        )


def test_predictor_registry_dispatches_new_class():
    assert "constant_predictor_for_test" in registered_names()
    p = build("constant_predictor_for_test", label="b")
    assert isinstance(p, ConstantPredictor)


def test_evaluator_handles_unknown_predictor_without_changes():
    p = build("constant_predictor_for_test", label="x")
    p.fit(["t1", "t2"], ["x", "x"], ["x", "y"])
    batch = p.predict(["t1", "t2", "t3"])
    evaluator = Evaluator(
        cost_options=[5.0],
        default_wrong_cost=5.0,
        human_triage_cost=1.0,
        correct_auto_cost=0.0,
        thresholds=[0.5],
    )
    result = evaluator.evaluate(
        predictor_name="constant_predictor_for_test",
        setting="trivial",
        batch=batch,
        true_labels=["x", "y", "x"],
        texts=["t1", "t2", "t3"],
        label_names=["x", "y"],
        confidence_estimator=ModelReportedConfidence(),
    )
    assert result.predictor_name == "constant_predictor_for_test"
    # 2 correct out of 3 -> accuracy 2/3.
    assert abs(result.classification["accuracy"] - 2 / 3) < 1e-9
    assert any(a["policy"] == "always_route" for a in result.abstention)
