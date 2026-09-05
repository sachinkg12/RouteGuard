"""End-to-end experiment runner.

Steps:
1. Load dataset and split (train / calibration / test).
2. Fit classical baseline on train.
3. For each enabled LLM experiment, build a PromptClassifier and predict on
   the (optionally subsampled) test set.
4. Score every predictor with each configured confidence method (no evaluator
   logic changes when methods/policies/models are added).
5. Emit result bundle (tables, figures, notes).

Usage:
    python scripts/run_classification.py --config configs/experiment_small_debug.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Load .env so provider keys are available before client construction.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from ticket_routing.abstention.threshold_policy import ThresholdAbstentionPolicy
from ticket_routing.confidence.agreement import AgreementConfidence
from ticket_routing.confidence.isotonic import IsotonicConfidence
from ticket_routing.confidence.model_reported import ModelReportedConfidence
from ticket_routing.confidence.self_consistency import SelfConsistencyConfidence
from ticket_routing.data.loaders import build_loader_from_config
from ticket_routing.data.splitters import (
    stratified_test_subsample_for_llm,
    stratified_three_way_split,
)
from ticket_routing.evaluation.evaluator import Evaluator
from ticket_routing.evaluation.metrics import bootstrap_difference
from ticket_routing.evaluation.cost import per_ticket_costs, CostModel
from ticket_routing.abstention.always_route import AlwaysRoutePolicy
from ticket_routing.llm.base import build_client_from_config
from ticket_routing.models.llm_prompt_classifier import LLMPromptClassifier
from ticket_routing.models.registry import build as build_predictor
from ticket_routing.models.tfidf_logreg import TfidfLogisticRegressionPredictor
# Side-effect imports register additional predictors with the registry.
from ticket_routing.models import tfidf_svm as _tfidf_svm  # noqa: F401
from ticket_routing.models import tfidf_rf as _tfidf_rf  # noqa: F401
# DistilBERT registration must be cheap and not import transformers eagerly.
# The class itself defers transformers/torch imports until fit()/predict().
from ticket_routing.models import distilbert_classifier as _distilbert  # noqa: F401
from ticket_routing.reporting.paper_bundle import build_paper_bundle
from ticket_routing.utils.config import dump_config, load_config
from ticket_routing.utils.logging import get_logger
from ticket_routing.utils.seeds import set_global_seed
from ticket_routing.utils.versions import collect_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    logger = get_logger("run_classification")
    cfg = load_config(args.config)
    if args.run_id:
        cfg.run_id = args.run_id

    set_global_seed(cfg.dataset.random_seed)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / f"{cfg.run_id}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "experiment_config.yaml")

    # 1. Load + split
    loader = build_loader_from_config(cfg.dataset)
    bundle = loader.load()
    logger.info(
        "Loaded dataset '%s' (n=%d, classes=%d).",
        bundle.metadata.get("dataset_name"),
        len(bundle),
        len(bundle.label_names),
    )
    split = stratified_three_way_split(
        bundle,
        train_fraction=cfg.split.train_fraction,
        calibration_fraction=cfg.split.calibration_fraction,
        test_fraction=cfg.split.test_fraction,
        seed=cfg.dataset.random_seed,
    )
    logger.info(
        "Split sizes: train=%d, calib=%d, test=%d",
        len(split.train_texts),
        len(split.calib_texts),
        len(split.test_texts),
    )

    # 2. Train classical baseline (always-on if enabled).
    predictors: list[tuple[str, str, object]] = []
    if cfg.classical.enabled:
        clf = TfidfLogisticRegressionPredictor(
            max_features=cfg.classical.max_features,
            ngram_max=cfg.classical.ngram_max,
            C=cfg.classical.C,
            random_state=cfg.dataset.random_seed,
        )
        clf.fit(split.train_texts, split.train_labels, split.label_names)
        predictors.append(("tfidf_logreg", "classical", clf))

    # 2b. Train any additional predictors built via the registry (SVM,
    # Random Forest, DistilBERT, ...). Purely additive: existing runs
    # without `extra_predictors:` in their YAML behave exactly as before.
    for extra in cfg.extra_predictors:
        if not extra.enabled:
            logger.info("Skipping disabled extra predictor: %s", extra.kind)
            continue
        params = dict(extra.params or {})
        params.setdefault("random_state", cfg.dataset.random_seed)
        extra_name = extra.name or extra.kind
        logger.info("Building extra predictor: %s (kind=%s)", extra_name, extra.kind)
        extra_clf = build_predictor(extra.kind, **params)
        extra_clf.fit(split.train_texts, split.train_labels, split.label_names)
        predictors.append((extra_name, "classical", extra_clf))

    # 3. Build & fit LLM predictors.
    for exp in cfg.llm_experiments:
        if not exp.enabled:
            logger.info("Skipping disabled LLM experiment: %s", exp.name)
            continue
        client = build_client_from_config(exp.client)
        prompt_version = exp.prompt_template_version or cfg.prompt_template_version
        llm = LLMPromptClassifier(
            client=client,
            setting=exp.setting,
            k_shot_per_class=exp.k_shot_per_class,
            prompt_template_version=prompt_version,
            self_consistency_samples=exp.self_consistency_samples,
            random_state=cfg.dataset.random_seed,
        )
        llm.fit(split.train_texts, split.train_labels, split.label_names)
        predictors.append((exp.name, exp.setting, llm))

    # 4. Predict on test split (LLMs use stratified subsample for cost control).
    full_test_texts, full_test_labels = split.test_texts, split.test_labels
    llm_test_texts, llm_test_labels = stratified_test_subsample_for_llm(
        full_test_texts,
        full_test_labels,
        max_total=cfg.dataset.max_test_samples_for_llm,
        seed=cfg.dataset.random_seed,
    )

    batches: dict[str, dict] = {}
    for name, setting, predictor in predictors:
        is_llm = isinstance(predictor, LLMPromptClassifier)
        eval_texts = llm_test_texts if is_llm else full_test_texts
        eval_labels = llm_test_labels if is_llm else full_test_labels
        logger.info("Predicting with %s/%s on %d test examples...", name, setting, len(eval_texts))
        batch = predictor.predict(eval_texts)
        batches[name] = {
            "predictor": predictor,
            "setting": setting,
            "batch": batch,
            "texts": eval_texts,
            "labels": eval_labels,
        }

    # 4b. Optionally pre-fit isotonic calibrators on the calibration split.
    prefit_isotonic: dict[str, IsotonicConfidence] = {}
    if "isotonic" in {m.lower() for m in cfg.confidence.methods}:
        llm_calib_texts, llm_calib_labels = stratified_test_subsample_for_llm(
            split.calib_texts,
            split.calib_labels,
            max_total=cfg.dataset.max_calib_samples_for_llm,
            seed=cfg.dataset.random_seed,
        )
        for name, setting, predictor in predictors:
            is_llm = isinstance(predictor, LLMPromptClassifier)
            calib_texts = llm_calib_texts if is_llm else split.calib_texts
            calib_labels = llm_calib_labels if is_llm else split.calib_labels
            logger.info(
                "Fitting isotonic calibrator for %s/%s on %d calibration tickets.",
                name,
                setting,
                len(calib_texts),
            )
            calib_batch = predictor.predict(calib_texts)
            raw_conf = calib_batch.confidence_scores or [0.0] * len(calib_texts)
            correctness = [
                int(p == t) for p, t in zip(calib_batch.predicted_labels, calib_labels)
            ]
            estimator = IsotonicConfidence()
            estimator.fit_from_pairs(raw_conf, correctness)
            prefit_isotonic[name] = estimator
            logger.info(
                "Isotonic for %s/%s fitted=%s on %d pairs.",
                name,
                setting,
                estimator.fitted,
                estimator.n_calibration_pairs,
            )

    # 5. Score every predictor with each configured confidence method.
    evaluator = Evaluator(
        cost_options=cfg.cost.wrong_auto_route_options,
        default_wrong_cost=cfg.cost.default_wrong_auto_route,
        human_triage_cost=cfg.cost.human_triage,
        correct_auto_cost=cfg.cost.correct_auto_route,
        thresholds=cfg.abstention.thresholds,
        agreement_min=cfg.abstention.agreement_min,
    )

    # For agreement confidence we need predictor sets aligned on the same examples.
    # We align on `llm_test_texts` if any LLM predictor exists; otherwise on full test.
    llm_names = [n for n, _, p in predictors if isinstance(p, LLMPromptClassifier)]
    if llm_names:
        # Re-run any non-LLM (classical) predictor on the LLM test subset so all
        # PredictionBatches are aligned for agreement-based confidence/policy.
        aligned_target_texts = llm_test_texts
        aligned_target_labels = llm_test_labels
        for name, info in batches.items():
            if not isinstance(info["predictor"], LLMPromptClassifier):
                logger.info(
                    "Re-predicting classical predictor %s on aligned LLM test subset for agreement.",
                    name,
                )
                info["batch_aligned"] = info["predictor"].predict(aligned_target_texts)
                info["texts_aligned"] = aligned_target_texts
                info["labels_aligned"] = aligned_target_labels
            else:
                info["batch_aligned"] = info["batch"]
                info["texts_aligned"] = info["texts"]
                info["labels_aligned"] = info["labels"]
    else:
        for info in batches.values():
            info["batch_aligned"] = info["batch"]
            info["texts_aligned"] = info["texts"]
            info["labels_aligned"] = info["labels"]

    results = []
    primary_order = list(batches.keys())
    for name in primary_order:
        info = batches[name]
        predictor = info["predictor"]
        setting = info["setting"]
        primary_batch = info["batch_aligned"]
        labels = info["labels_aligned"]
        texts = info["texts_aligned"]
        auxiliary = [
            batches[other]["batch_aligned"]
            for other in primary_order
            if other != name
        ]

        for method in cfg.confidence.methods:
            estimator = _build_confidence(
                method,
                predictor,
                prefit_isotonic=prefit_isotonic.get(name),
            )
            result = evaluator.evaluate(
                predictor_name=name,
                setting=setting,
                batch=primary_batch,
                true_labels=labels,
                texts=texts,
                label_names=split.label_names,
                confidence_estimator=estimator,
                auxiliary_batches=auxiliary or None,
            )
            results.append(result)
            logger.info(
                "Evaluated %s/%s with confidence=%s: acc=%.4f macro_F1=%.4f ECE=%.4f",
                name,
                setting,
                method,
                result.classification["accuracy"],
                result.classification["macro_f1"],
                result.calibration["ece"],
            )

    # Optional bootstrap CI for the headline cost comparison.
    if cfg.bootstrap.enabled and results:
        _attach_bootstrap_cis(results, cfg)

    # 6. Result bundle.
    env = collect_environment()
    artifacts = build_paper_bundle(
        out_dir=run_dir,
        cfg=cfg,
        dataset_bundle=bundle,
        split_bundle=split,
        results=results,
        env=env,
    )
    logger.info("Wrote %d result-bundle artifacts under %s", len(artifacts), run_dir)
    logger.info("Run complete.")


def _build_confidence(
    method: str,
    predictor,
    *,
    prefit_isotonic: IsotonicConfidence | None = None,
):
    method = method.lower()
    if method == "model_reported":
        return ModelReportedConfidence()
    if method == "agreement":
        return AgreementConfidence()
    if method == "self_consistency":
        return SelfConsistencyConfidence(predictor)
    if method == "isotonic":
        # Falls back to a pass-through (raw confidence) when the runner
        # could not pre-fit (e.g. degenerate calibration split, or this
        # predictor was missing at fit time).
        return prefit_isotonic if prefit_isotonic is not None else IsotonicConfidence()
    raise ValueError(f"Unknown confidence method: {method}")


def _attach_bootstrap_cis(results, cfg) -> None:
    """Attach a paired bootstrap CI for best-threshold vs always-route per result."""
    cost = CostModel(
        correct_auto_route=cfg.cost.correct_auto_route,
        human_triage=cfg.cost.human_triage,
        wrong_auto_route=cfg.cost.default_wrong_auto_route,
    )
    for r in results:
        # Find best threshold policy by expected cost reduction.
        threshold_rows = [c for c in r.cost if c["policy"].startswith("threshold@") and c["wrong_auto_route_cost"] == cost.wrong_auto_route]
        if not threshold_rows:
            continue
        best = min(threshold_rows, key=lambda c: c["expected_cost_per_ticket"])
        baseline = next(
            (c for c in r.cost if c["policy"] == "always_route" and c["wrong_auto_route_cost"] == cost.wrong_auto_route),
            None,
        )
        if not baseline:
            continue
        ci = bootstrap_difference(
            best["per_ticket_costs"],
            baseline["per_ticket_costs"],
            n_samples=cfg.bootstrap.samples,
            seed=cfg.bootstrap.seed,
        )
        # Record WHICH row this CI belongs to, so the reporting layer attaches it
        # to exactly that (policy, wrong_route_cost) cell rather than broadcasting.
        ci["best_policy"] = best["policy"]
        ci["wrong_route_cost"] = cost.wrong_auto_route
        r.metadata.setdefault("bootstrap", {})["best_threshold_vs_always_route_cost_per_ticket"] = ci


if __name__ == "__main__":
    main()
