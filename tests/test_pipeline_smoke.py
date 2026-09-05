"""Smoke test: tiny end-to-end run produces a result bundle."""
from __future__ import annotations

from pathlib import Path

from ticket_routing.confidence.model_reported import ModelReportedConfidence
from ticket_routing.data.loaders import SyntheticTicketLoader
from ticket_routing.data.splitters import stratified_three_way_split
from ticket_routing.evaluation.evaluator import Evaluator
from ticket_routing.models.tfidf_logreg import TfidfLogisticRegressionPredictor
from ticket_routing.reporting.paper_bundle import build_paper_bundle
from ticket_routing.utils.config import ExperimentConfig, DatasetConfig
from ticket_routing.utils.versions import collect_environment


def test_paper_bundle_smoke(tmp_path: Path):
    bundle = SyntheticTicketLoader(n_per_class=30, seed=7).load()
    split = stratified_three_way_split(bundle, 0.7, 0.1, 0.2, seed=7)

    clf = TfidfLogisticRegressionPredictor(max_features=2000, ngram_max=1, C=1.0)
    clf.fit(split.train_texts, split.train_labels, split.label_names)
    batch = clf.predict(split.test_texts)

    evaluator = Evaluator(
        cost_options=[2.0, 5.0],
        default_wrong_cost=5.0,
        human_triage_cost=1.0,
        correct_auto_cost=0.0,
        thresholds=[0.5, 0.7],
    )
    result = evaluator.evaluate(
        predictor_name="tfidf_logreg",
        setting="classical",
        batch=batch,
        true_labels=split.test_labels,
        texts=split.test_texts,
        label_names=split.label_names,
        confidence_estimator=ModelReportedConfidence(),
    )

    cfg = ExperimentConfig(dataset=DatasetConfig(dataset_name="synthetic"))
    cfg.cost.default_wrong_auto_route = 5.0
    cfg.run_id = "smoke"
    artifacts = build_paper_bundle(
        out_dir=tmp_path,
        cfg=cfg,
        dataset_bundle=bundle,
        split_bundle=split,
        results=[result],
        env=collect_environment(),
    )
    bundle_dir = tmp_path / "paper_bundle"
    assert (bundle_dir / "paper_tables.md").exists()
    assert (bundle_dir / "result_summary.md").exists()
    assert (bundle_dir / "tables" / "table_2_classification.csv").exists()
    assert (bundle_dir / "figures" / "figure_2_class_distribution.png").exists()
    assert len(artifacts) > 5
