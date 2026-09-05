"""Evaluator: composes metrics, calibration, abstention, and cost over predictors.

The evaluator treats every predictor as a Predictor + PredictionBatch. It does
not inspect predictor internals, and stays unchanged when you add new models,
confidence methods, or abstention policies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .calibration import correctness_vector, expected_calibration_error
from .cost import CostModel, cost_table_row, expected_cost_per_ticket, per_ticket_costs
from .error_analysis import top_confused_pairs
from .metrics import compute_abstention_metrics, compute_classification_metrics
from ..abstention.always_route import AlwaysRoutePolicy
from ..abstention.base import AUTO_ROUTE, DEFER_TO_HUMAN, AbstentionPolicy
from ..abstention.threshold_policy import ThresholdAbstentionPolicy
from ..confidence.base import ConfidenceEstimator
from ..confidence.model_reported import ModelReportedConfidence
from ..models.base import PredictionBatch


@dataclass
class PredictorResult:
    predictor_name: str
    setting: str
    classification: Dict = field(default_factory=dict)
    calibration: Dict = field(default_factory=dict)
    abstention: List[Dict] = field(default_factory=list)
    cost: List[Dict] = field(default_factory=list)
    error_analysis: List[Dict] = field(default_factory=list)
    confidence_method: str = "model_reported"
    confidences: List[float] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    parse_status: List[str] = field(default_factory=list)
    per_ticket_costs_baseline: List[float] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class Evaluator:
    """Runs all metrics for a single predictor + confidence method combo."""

    def __init__(
        self,
        cost_options: Sequence[float],
        default_wrong_cost: float,
        human_triage_cost: float,
        correct_auto_cost: float,
        thresholds: Sequence[float],
        agreement_min: int = 2,
    ) -> None:
        self.cost_options = list(cost_options)
        self.default_wrong_cost = default_wrong_cost
        self.human_triage_cost = human_triage_cost
        self.correct_auto_cost = correct_auto_cost
        self.thresholds = list(thresholds)
        self.agreement_min = agreement_min

    def evaluate(
        self,
        predictor_name: str,
        setting: str,
        batch: PredictionBatch,
        true_labels: Sequence[str],
        texts: Sequence[str],
        label_names: Sequence[str],
        confidence_estimator: ConfidenceEstimator,
        auxiliary_batches: Optional[Sequence[PredictionBatch]] = None,
    ) -> PredictorResult:
        confidences = confidence_estimator.score(batch, auxiliary_batches)

        # Classification
        cls_metrics = compute_classification_metrics(batch, true_labels, label_names)

        # Calibration (uses correctness vector)
        correctness = correctness_vector(batch.predicted_labels, true_labels)
        ece = expected_calibration_error(confidences, correctness)
        calib = {
            "ece": ece["ece"],
            "n_bins": ece["n_bins"],
            "bins": ece["bins"],
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "accuracy": cls_metrics["accuracy"],
        }

        # Abstention (always_route + thresholds + agreement if aux available)
        cost_default = CostModel(
            correct_auto_route=self.correct_auto_cost,
            human_triage=self.human_triage_cost,
            wrong_auto_route=self.default_wrong_cost,
        )
        always_route = AlwaysRoutePolicy()
        always_route_decisions = always_route.decide(batch, confidences, auxiliary_batches)
        baseline_always_route_costs = per_ticket_costs(
            batch, true_labels, always_route_decisions, cost_default
        )
        baseline_always_defer_e_cost = self.human_triage_cost  # by definition

        abstention_results: List[Dict] = []
        cost_results: List[Dict] = []

        for policy in self._build_policies(auxiliary_batches):
            decisions = policy.decide(batch, confidences, auxiliary_batches)
            abst_metrics = compute_abstention_metrics(batch, true_labels, decisions)
            abst_metrics["policy"] = policy.name
            abstention_results.append(abst_metrics)

            # Cost sensitivity sweep across configured wrong-route options.
            for wrong_cost in self.cost_options:
                cm = CostModel(
                    correct_auto_route=self.correct_auto_cost,
                    human_triage=self.human_triage_cost,
                    wrong_auto_route=wrong_cost,
                )
                base_ar_costs = per_ticket_costs(batch, true_labels, always_route_decisions, cm)
                row = cost_table_row(
                    batch=batch,
                    true_labels=true_labels,
                    decisions=decisions,
                    cost=cm,
                    baseline_always_route_costs=base_ar_costs,
                    baseline_always_defer_cost_per_ticket=self.human_triage_cost,
                )
                row["policy"] = policy.name
                cost_results.append(row)

        error_pairs = top_confused_pairs(batch, true_labels, texts, k=5)

        return PredictorResult(
            predictor_name=predictor_name,
            setting=setting,
            classification=cls_metrics,
            calibration=calib,
            abstention=abstention_results,
            cost=cost_results,
            error_analysis=error_pairs,
            confidence_method=confidence_estimator.name,
            confidences=list(confidences),
            predictions=list(batch.predicted_labels),
            parse_status=list(batch.parse_status or []),
            per_ticket_costs_baseline=baseline_always_route_costs,
            metadata=dict(batch.model_metadata),
        )

    # ----- private --------------------------------------------------------

    def _build_policies(
        self,
        auxiliary_batches: Optional[Sequence[PredictionBatch]],
    ) -> List[AbstentionPolicy]:
        policies: List[AbstentionPolicy] = [AlwaysRoutePolicy()]
        for t in self.thresholds:
            policies.append(ThresholdAbstentionPolicy(threshold=t))
        if auxiliary_batches:
            from ..abstention.agreement_policy import AgreementAbstentionPolicy

            total = 1 + len(auxiliary_batches)
            # Add (agreement_min)-of-N and N-of-N agreement policies.
            policies.append(AgreementAbstentionPolicy(min_agree=self.agreement_min))
            if total >= 3:
                policies.append(AgreementAbstentionPolicy(min_agree=total))
        return policies
