"""Generate the audited numerical source of truth for the ICTAI camera-ready.

This script makes no network or LLM calls. It retrains the three inexpensive
classical models from the frozen dataset/split, reads the saved DistilBERT
calibration+test predictions, and re-analyses the frozen May 2026 LLM test
artifacts. All policy selection is either performed on calibration or fixed
analytically before looking at test outcomes.

Run from the repository root:

    .venv/bin/python scripts/generate_camera_ready_manifest.py \
      --paper-dir /absolute/path/to/papers/ictai2026
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ticket_routing.data.loaders import build_loader_from_config
from ticket_routing.data.splitters import stratified_three_way_split
from ticket_routing.evaluation.calibration import expected_calibration_error
from ticket_routing.evaluation.cost import CostModel
from ticket_routing.evaluation.metrics import (
    bootstrap_difference,
    bootstrap_relative_reduction,
)
from ticket_routing.evaluation.selection import select_threshold_on_calibration
from ticket_routing.evaluation.selective import correctness_auroc, tie_group_aurc
from ticket_routing.models.base import PARSE_OK, PredictionBatch
from ticket_routing.models.tfidf_logreg import TfidfLogisticRegressionPredictor
from ticket_routing.models.tfidf_rf import TfidfRandomForestPredictor
from ticket_routing.models.tfidf_svm import TfidfLinearSVMPredictor
from ticket_routing.utils.config import load_config
from ticket_routing.utils.hashing import sha256_iter_strings


COST = CostModel(correct_auto_route=0.0, human_triage=1.0, wrong_auto_route=5.0)
PREDECLARED_GRID = [0.50, 0.60, 0.70, 0.80, 0.90]
DENSE_GRID = [round(v, 2) for v in np.arange(0.00, 1.001, 0.01)]
DISTIL_DENSE_GRID = [round(v, 2) for v in np.arange(0.50, 1.00, 0.01)]
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 1337
BREAK_EVEN_THRESHOLD = 1.0 - COST.human_triage / COST.wrong_auto_route
REVIEW_SOURCE_SHA256 = "292cea4e68d502b78a90c7f8e69311400a3a56ad64317448306951fc59c84a7b"
REVIEW_SUPPLEMENT_SHA256 = "0a6f50688c1330617a89e4548222d786240f220ad7cdef5e40c8d8d80b28baba"

LLM_BUNDLE = (
    REPO_ROOT
    / "artifacts"
    / "ictai2026"
    / "llm_inputs"
)
TABLE_II_INPUT = REPO_ROOT / "artifacts" / "ictai2026" / "table_ii_common_subset.json"
DISTIL_PREDICTIONS = REPO_ROOT / "outputs" / "distilbert_calib_test_preds.json"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment_default.yaml"
CLASSICAL_CONFIG = REPO_ROOT / "configs" / "experiment_classical_extras.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_archive_record(
    path: Path, expected_sha256: str, *, manifest_path: str | None = None
) -> dict:
    """Describe a private review-time archive without requiring redistribution."""
    if not path.exists():
        return {
            "path": manifest_path or path.name,
            "sha256": expected_sha256,
            "availability": "not_redistributed; expected historical hash retained",
        }
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"historical archive hash mismatch for {path}: "
            f"expected {expected_sha256}, found {actual}"
        )
    return {
        "path": manifest_path or path.name,
        "sha256": actual,
        "availability": "present_and_verified",
    }


def batch_for(predictions: Sequence[str]) -> PredictionBatch:
    return PredictionBatch(
        predicted_labels=list(predictions),
        parse_status=[PARSE_OK] * len(predictions),
    )


def policy_metrics(
    correctness: Sequence[bool],
    confidences: Sequence[float],
    threshold: float,
) -> dict:
    correct = np.asarray(correctness, dtype=bool)
    confidence = np.asarray(confidences, dtype=float)
    routed = confidence >= threshold
    policy_costs = np.where(
        routed,
        np.where(correct, COST.correct_auto_route, COST.wrong_auto_route),
        COST.human_triage,
    ).astype(float)
    always_route = np.where(
        correct, COST.correct_auto_route, COST.wrong_auto_route
    ).astype(float)
    always_defer = np.full(correct.size, COST.human_triage, dtype=float)
    n_routed = int(routed.sum())

    rel_route = bootstrap_relative_reduction(
        policy_costs,
        always_route,
        n_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    rel_defer = bootstrap_relative_reduction(
        policy_costs,
        always_defer,
        n_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    diff_route = bootstrap_difference(
        policy_costs,
        always_route,
        n_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    diff_defer = bootstrap_difference(
        policy_costs,
        always_defer,
        n_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "threshold": float(threshold),
        "n_test": int(correct.size),
        "n_routed": n_routed,
        "n_deferred": int(correct.size - n_routed),
        "coverage": float(routed.mean()),
        "accuracy_on_routed": float(correct[routed].mean()) if n_routed else None,
        "wrong_route_rate": float(np.mean(routed & ~correct)),
        "always_route_wrong_rate": float(np.mean(~correct)),
        "always_route_cost": float(always_route.mean()),
        "always_defer_cost": float(always_defer.mean()),
        "policy_cost": float(policy_costs.mean()),
        "relative_reduction_vs_always_route": rel_route,
        "relative_reduction_vs_always_defer": rel_defer,
        "cost_difference_vs_always_route": diff_route,
        "cost_difference_vs_always_defer": diff_defer,
    }


def select_and_evaluate(
    *,
    calibration_predictions: Sequence[str],
    calibration_labels: Sequence[str],
    calibration_scores: Sequence[float],
    test_predictions: Sequence[str],
    test_labels: Sequence[str],
    test_scores: Sequence[float],
    threshold_grid: Sequence[float],
) -> dict:
    selection = select_threshold_on_calibration(
        batch=batch_for(calibration_predictions),
        true_labels=calibration_labels,
        confidences=calibration_scores,
        thresholds=threshold_grid,
        cost=COST,
        selection_split="calibration",
    )
    correctness = [p == y for p, y in zip(test_predictions, test_labels)]
    return {
        "selection": selection.to_metadata(),
        "test": policy_metrics(correctness, test_scores, selection.threshold),
        "test_accuracy": float(np.mean(correctness)),
        "test_macro_f1": float(
            f1_score(
                test_labels,
                test_predictions,
                labels=sorted(set(test_labels)),
                average="macro",
                zero_division=0,
            )
        ),
        "test_ece_10_equal_width_bins": float(
            expected_calibration_error(test_scores, correctness, n_bins=10)["ece"]
        ),
    }


def classical_analysis(split) -> tuple[list[dict], list[dict]]:
    models = [
        (
            "TF-IDF + logistic regression",
            TfidfLogisticRegressionPredictor(
                max_features=100_000, ngram_max=2, C=1.0, random_state=42
            ),
        ),
        (
            "TF-IDF + calibrated linear SVM",
            TfidfLinearSVMPredictor(
                max_features=100_000,
                ngram_max=2,
                C=1.0,
                random_state=42,
                calibration_cv=3,
            ),
        ),
        (
            "TF-IDF + random forest",
            TfidfRandomForestPredictor(
                max_features=50_000,
                ngram_max=2,
                n_estimators=300,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
    submitted_expectations = {
        "TF-IDF + logistic regression": {"threshold": 0.60, "reduction": 0.3808},
        "TF-IDF + calibrated linear SVM": {"threshold": 0.70, "reduction": 0.3674},
        "TF-IDF + random forest": {"threshold": 0.60, "reduction": 0.3730},
    }

    primary = []
    comparator_rows = []
    for name, model in models:
        print(f"Fitting {name}...", flush=True)
        model.fit(split.train_texts, split.train_labels, split.label_names)
        calibration_batch = model.predict(split.calib_texts)
        test_batch = model.predict(split.test_texts)
        record = select_and_evaluate(
            calibration_predictions=calibration_batch.predicted_labels,
            calibration_labels=split.calib_labels,
            calibration_scores=calibration_batch.confidence_scores or [],
            test_predictions=test_batch.predicted_labels,
            test_labels=split.test_labels,
            test_scores=test_batch.confidence_scores or [],
            threshold_grid=PREDECLARED_GRID,
        )
        record["model"] = name
        record["confidence_score"] = "top_class_probability"
        expected = submitted_expectations[name]
        actual_reduction = record["test"]["relative_reduction_vs_always_route"][
            "point_estimate"
        ]
        record["submitted_rounding_check"] = {
            "expected_threshold": expected["threshold"],
            "expected_relative_reduction": expected["reduction"],
            "passes": bool(
                record["selection"]["threshold"] == expected["threshold"]
                and abs(actual_reduction - expected["reduction"]) <= 0.0005
            ),
        }
        primary.append(record)

        if name == "TF-IDF + logistic regression":
            calibration_proba = model.predict_proba_full(split.calib_texts)
            test_proba = model.predict_proba_full(split.test_texts)
            comparator_scores = {
                "top_class_probability": (
                    np.max(calibration_proba, axis=1),
                    np.max(test_proba, axis=1),
                ),
                "top_two_probability_margin": (
                    _top_two_margin(calibration_proba),
                    _top_two_margin(test_proba),
                ),
                "normalized_inverse_entropy": (
                    _normalized_inverse_entropy(calibration_proba),
                    _normalized_inverse_entropy(test_proba),
                ),
            }
            for score_name, (cal_score, test_score) in comparator_scores.items():
                comparison = select_and_evaluate(
                    calibration_predictions=calibration_batch.predicted_labels,
                    calibration_labels=split.calib_labels,
                    calibration_scores=cal_score,
                    test_predictions=test_batch.predicted_labels,
                    test_labels=split.test_labels,
                    test_scores=test_score,
                    threshold_grid=DENSE_GRID,
                )
                comparison["model"] = name
                comparison["confidence_score"] = score_name
                comparison["analysis_role"] = (
                    "nontrivial_rejection_comparator"
                    if score_name != "top_class_probability"
                    else "dense_grid_probability_reference"
                )
                comparator_rows.append(comparison)

    return primary, comparator_rows


def distilbert_analysis() -> tuple[dict, dict]:
    data = json.loads(DISTIL_PREDICTIONS.read_text())
    primary = select_and_evaluate(
        calibration_predictions=data["calib_predictions"],
        calibration_labels=data["calib_labels"],
        calibration_scores=data["calib_confidences"],
        test_predictions=data["test_predictions"],
        test_labels=data["test_labels"],
        test_scores=data["test_confidences"],
        threshold_grid=PREDECLARED_GRID,
    )
    primary["model"] = "fine-tuned DistilBERT"
    primary["confidence_score"] = "top_class_probability"
    actual_reduction = primary["test"]["relative_reduction_vs_always_route"][
        "point_estimate"
    ]
    primary["submitted_rounding_check"] = {
        "expected_threshold": 0.90,
        "expected_relative_reduction": 0.355,
        "passes": bool(
            primary["selection"]["threshold"] == 0.90
            and abs(actual_reduction - 0.355) <= 0.0005
        ),
    }

    dense = select_and_evaluate(
        calibration_predictions=data["calib_predictions"],
        calibration_labels=data["calib_labels"],
        calibration_scores=data["calib_confidences"],
        test_predictions=data["test_predictions"],
        test_labels=data["test_labels"],
        test_scores=data["test_confidences"],
        threshold_grid=DISTIL_DENSE_GRID,
    )
    dense["model"] = "fine-tuned DistilBERT"
    dense["confidence_score"] = "top_class_probability"
    dense["analysis_role"] = "post_hoc_grid_density_sensitivity"
    return primary, dense


def llm_analysis() -> tuple[list[dict], list[dict], dict]:
    raw_paths = sorted(LLM_BUNDLE.glob("*__model_reported.json"))
    raw_paths = [p for p in raw_paths if not p.name.startswith("tfidf_logreg")]
    if len(raw_paths) != 8:
        raise RuntimeError(f"expected 8 LLM conditions, found {len(raw_paths)}")

    raw_records = {model_key(path): json.loads(path.read_text()) for path in raw_paths}
    model_keys = sorted(raw_records)
    lengths = {len(record["predictions"]) for record in raw_records.values()}
    if lengths != {1000}:
        raise RuntimeError(f"LLM artifacts are not the expected aligned n=1000: {lengths}")

    all_llm_predictions = {
        key: np.asarray(record["predictions"], dtype=object)
        for key, record in raw_records.items()
    }
    classical = json.loads((LLM_BUNDLE / "tfidf_logreg__classical__model_reported.json").read_text())
    classical_predictions = np.asarray(classical["predictions"], dtype=object)
    common_subset = json.loads(TABLE_II_INPUT.read_text())
    true_labels = np.asarray(common_subset["true_labels"], dtype=object)
    projected_classical_predictions = np.asarray(
        common_subset["models"]["tfidf_logreg"]["predictions"], dtype=object
    )
    if true_labels.size != 1000:
        raise RuntimeError(
            f"Table-II label projection is not the expected aligned n=1000: {true_labels.size}"
        )
    if not np.array_equal(classical_predictions, projected_classical_predictions):
        raise RuntimeError(
            "Table-II labels cannot be bound to LLM rows: classical prediction order differs"
        )
    label_order = sorted(set(true_labels.tolist()))

    confidence_rows = []
    fixed_policy_rows = []
    m9_validation = {}
    for key in model_keys:
        raw = raw_records[key]
        prefix = raw_paths_by_key(raw_paths)[key].name.removesuffix("__model_reported.json")
        isotonic = json.loads((LLM_BUNDLE / f"{prefix}__isotonic.json").read_text())
        agreement = json.loads((LLM_BUNDLE / f"{prefix}__agreement.json").read_text())
        if not (
            raw["predictions"] == isotonic["predictions"] == agreement["predictions"]
        ):
            raise RuntimeError(f"prediction order mismatch across confidence methods: {key}")

        correctness = correctness_from_raw_result(raw)
        label_correctness = (
            np.asarray(raw["predictions"], dtype=object) == true_labels
        ).astype(int)
        if not np.array_equal(correctness, label_correctness):
            raise RuntimeError(
                f"Table-II labels disagree with the saved LLM correctness vector: {key}"
            )
        macro_f1 = float(
            f1_score(
                true_labels,
                raw["predictions"],
                labels=label_order,
                average="macro",
                zero_division=0,
            )
        )
        primary_predictions = all_llm_predictions[key]
        m8 = np.mean(
            np.vstack(
                [predictions == primary_predictions for predictions in all_llm_predictions.values()]
            ),
            axis=0,
        )
        m9 = (
            np.sum(
                np.vstack(
                    [predictions == primary_predictions for predictions in all_llm_predictions.values()]
                ),
                axis=0,
            )
            + (classical_predictions == primary_predictions).astype(int)
        ) / 9.0
        saved_m9 = np.asarray(agreement["confidences"], dtype=float)
        matches_saved = bool(np.allclose(m9, saved_m9, atol=1e-12))
        m9_validation[key] = matches_saved
        if not matches_saved:
            raise RuntimeError(f"reconstructed M=9 agreement differs from saved scores: {key}")

        score_sets = {
            "raw_model_reported": np.asarray(raw["confidences"], dtype=float),
            "isotonic": np.asarray(isotonic["confidences"], dtype=float),
            "agreement_m9_mixed": saved_m9,
            "agreement_m8_llm_only": m8,
        }
        raw_ece = expected_calibration_error(
            score_sets["raw_model_reported"], correctness
        )["ece"]
        raw_auroc = correctness_auroc(score_sets["raw_model_reported"], correctness)
        for method, scores in score_sets.items():
            ece = expected_calibration_error(scores, correctness)["ece"]
            selective = tie_group_aurc(scores, correctness)
            confidence_rows.append(
                {
                    "model_condition": key,
                    "confidence_method": method,
                    "n_test": int(len(correctness)),
                    "accuracy": float(np.mean(correctness)),
                    "macro_f1": macro_f1,
                    "ece_10_equal_width_bins": float(ece),
                    "relative_ece_reduction_vs_raw": (
                        float(1.0 - ece / raw_ece) if raw_ece else None
                    ),
                    "correctness_auroc": correctness_auroc(scores, correctness),
                    "correctness_auroc_delta_vs_raw": float(
                        correctness_auroc(scores, correctness) - raw_auroc
                    ),
                    "tie_group_aurc": selective["aurc"],
                    "n_unique_scores": selective["n_unique_scores"],
                    "tie_handling": selective["tie_handling"],
                    "score_min": float(np.min(scores)),
                    "score_max": float(np.max(scores)),
                    "source": (
                        "formally reconstructed from the eight frozen LLM prediction vectors"
                        if method == "agreement_m8_llm_only"
                        else "frozen May 2026 per-ticket artifact"
                    ),
                }
            )

        fixed = policy_metrics(
            correctness,
            score_sets["isotonic"],
            BREAK_EVEN_THRESHOLD,
        )
        fixed_policy_rows.append(
            {
                "model_condition": key,
                "confidence_method": "isotonic",
                "policy_rule": "auto-route iff isotonic P(correct) >= 0.80",
                "selection_basis": (
                    "analytic break-even under (correct=0, triage=1, wrong=5); "
                    "not selected on test"
                ),
                **fixed,
            }
        )

    return confidence_rows, fixed_policy_rows, m9_validation


def correctness_from_raw_result(record: dict) -> np.ndarray:
    rows = [
        row
        for row in record["cost"]
        if row["policy"] == "always_route" and row["wrong_auto_route_cost"] == 5.0
    ]
    if len(rows) != 1:
        raise RuntimeError("cannot identify the canonical always-route cost vector")
    costs = np.asarray(rows[0]["per_ticket_costs"], dtype=float)
    if not np.all(np.isin(costs, [0.0, 5.0])):
        raise RuntimeError("unexpected values in always-route per-ticket costs")
    return (costs == 0.0).astype(int)


def model_key(path: Path) -> str:
    return path.name.removesuffix("__model_reported.json")


def raw_paths_by_key(paths: Iterable[Path]) -> dict[str, Path]:
    return {model_key(path): path for path in paths}


def _top_two_margin(probabilities: np.ndarray) -> np.ndarray:
    top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
    return np.max(top_two, axis=1) - np.min(top_two, axis=1)


def _normalized_inverse_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return 1.0 - entropy / math.log(probabilities.shape[1])


def flatten_policy_record(record: dict) -> dict:
    test = record["test"]
    reduction = test["relative_reduction_vs_always_route"]
    return {
        "model": record["model"],
        "confidence_score": record["confidence_score"],
        "selection_split": record["selection"]["selection_split"],
        "threshold_grid": json.dumps(record["selection"]["threshold_grid"]),
        "selected_threshold": record["selection"]["threshold"],
        "n_test": test["n_test"],
        "test_accuracy": record["test_accuracy"],
        "test_macro_f1": record["test_macro_f1"],
        "test_ece_10_equal_width_bins": record["test_ece_10_equal_width_bins"],
        "always_route_cost": test["always_route_cost"],
        "policy_cost": test["policy_cost"],
        "relative_reduction": reduction["point_estimate"],
        "relative_reduction_ci_low": reduction["ci_low"],
        "relative_reduction_ci_high": reduction["ci_high"],
        "n_routed": test["n_routed"],
        "coverage": test["coverage"],
        "accuracy_on_routed": test["accuracy_on_routed"],
        "wrong_route_rate": test["wrong_route_rate"],
        "analysis_role": record.get("analysis_role", "primary_predeclared_grid"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_audit_report(path: Path, manifest: dict) -> None:
    classical = manifest["classical_predeclared_grid"]
    lr = next(row for row in classical if row["model"] == "TF-IDF + logistic regression")
    distil_dense = manifest["distilbert_dense_grid_sensitivity"]
    gpt_few_iso = next(
        row
        for row in manifest["llm_confidence_metrics"]
        if row["model_condition"].startswith("gpt4o_mini_few_shot")
        and row["confidence_method"] == "isotonic"
    )
    haiku_fixed = next(
        row
        for row in manifest["llm_fixed_break_even_policy"]
        if row["model_condition"].startswith("haiku_few_shot")
    )
    report = f"""# Camera-Ready Numerical Audit Report

Generated exclusively from `claim_manifest.json`. Do not hand-copy values from
older tables or exploratory scripts.

## Results that survive the audit

- TF-IDF + logistic regression: threshold {lr['selection']['threshold']:.2f}
  selected on calibration; test cost {lr['test']['always_route_cost']:.4f} to
  {lr['test']['policy_cost']:.4f}; relative reduction
  {100 * lr['test']['relative_reduction_vs_always_route']['point_estimate']:.2f}%
  (ratio-bootstrap 95% CI
  [{100 * lr['test']['relative_reduction_vs_always_route']['ci_low']:.2f}%,
  {100 * lr['test']['relative_reduction_vs_always_route']['ci_high']:.2f}%]).
- Its wrong-route rate changes from {100 * lr['test']['always_route_wrong_rate']:.2f}%
  to {100 * lr['test']['wrong_route_rate']:.2f}% while routing
  {lr['test']['n_routed']}/{lr['test']['n_test']} tickets.
- DistilBERT's post-hoc dense-grid sensitivity selects
  {distil_dense['selection']['threshold']:.2f} on calibration and produces a
  {100 * distil_dense['test']['relative_reduction_vs_always_route']['point_estimate']:.2f}%
  test reduction, supporting rather than replacing the predeclared-grid result.

## Reviewer concerns confirmed by the audit

- GPT-4o-mini few-shot isotonic confidence has
  correctness AUROC {gpt_few_iso['correctness_auroc']:.3f} with
  {gpt_few_iso['n_unique_scores']} unique score. Its low ECE therefore does not
  show useful correctness ranking.
- The LLM-only M=8 agreement control is reported separately from the mixed M=9
  pool; the latter includes the supervised TF-IDF + LR prediction.
- LLM deployment policy results use the analytic break-even threshold 0.80, not
  a threshold chosen on test. Haiku few-shot routes only
  {haiku_fixed['n_routed']}/{haiku_fixed['n_test']} tickets and is explicitly a
  low-coverage observation.
- Top-two margin and normalized inverse entropy are now evaluated as non-trivial
  rejection comparators with their thresholds selected on calibration.

## Interpretation rule

The audited manifest controls the camera-ready text. Any claim that conflicts
with it must be corrected or removed. All cost conclusions are limited to this
dataset and the illustrative (0,1,5) cost model.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", required=True, type=Path)
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    output_dir = paper_dir / "camera_ready_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(DEFAULT_CONFIG)
    bundle = build_loader_from_config(cfg.dataset).load()
    split = stratified_three_way_split(
        bundle,
        train_fraction=cfg.split.train_fraction,
        calibration_fraction=cfg.split.calibration_fraction,
        test_fraction=cfg.split.test_fraction,
        seed=cfg.dataset.random_seed,
    )
    split_hashes = {
        "train": sha256_iter_strings(
            f"{text}\t{label}" for text, label in zip(split.train_texts, split.train_labels)
        ),
        "calibration": sha256_iter_strings(
            f"{text}\t{label}" for text, label in zip(split.calib_texts, split.calib_labels)
        ),
        "test": sha256_iter_strings(
            f"{text}\t{label}" for text, label in zip(split.test_texts, split.test_labels)
        ),
    }

    classical, comparators = classical_analysis(split)
    distil_primary, distil_dense = distilbert_analysis()
    classical.append(distil_primary)
    llm_confidence, llm_fixed, m9_validation = llm_analysis()

    source_zip = (
        paper_dir
        / "private_review_time_archives"
        / "routeguard_ictai2026_overleaf.zip"
    )
    supplement_zip = (
        paper_dir
        / "private_review_time_archives"
        / "routeguard_supplementary.zip"
    )
    manifest = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "scope": {
            "dataset": "IT Service Ticket Classification Dataset",
            "n_examples": len(bundle),
            "n_classes": len(bundle.label_names),
            "cost_model": {
                "correct_auto_route": COST.correct_auto_route,
                "human_triage": COST.human_triage,
                "wrong_auto_route": COST.wrong_auto_route,
                "status": "illustrative; organizations must estimate local costs",
            },
            "bootstrap": {
                "method": "paired resampling of tickets with the reported ratio recomputed inside each resample",
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "interval": "percentile 95%",
            },
        },
        "inputs": {
            "dataset": {"path": str(cfg.dataset.dataset_path), "sha256": sha256_file(REPO_ROOT / cfg.dataset.dataset_path)},
            "default_config": {"path": str(DEFAULT_CONFIG.relative_to(REPO_ROOT)), "sha256": sha256_file(DEFAULT_CONFIG)},
            "classical_config": {"path": str(CLASSICAL_CONFIG.relative_to(REPO_ROOT)), "sha256": sha256_file(CLASSICAL_CONFIG)},
            "distilbert_predictions": {"path": str(DISTIL_PREDICTIONS.relative_to(REPO_ROOT)), "sha256": sha256_file(DISTIL_PREDICTIONS)},
            "llm_common_labels": {"path": str(TABLE_II_INPUT.relative_to(REPO_ROOT)), "sha256": sha256_file(TABLE_II_INPUT)},
            "llm_bundle_files": {
                str(path.relative_to(REPO_ROOT)): sha256_file(path)
                for path in sorted(LLM_BUNDLE.glob("*.json"))
            },
            "submitted_source_zip": historical_archive_record(
                source_zip,
                REVIEW_SOURCE_SHA256,
                manifest_path=source_zip.relative_to(paper_dir).as_posix(),
            ),
            "submitted_supplement_zip": historical_archive_record(
                supplement_zip,
                REVIEW_SUPPLEMENT_SHA256,
                manifest_path=supplement_zip.relative_to(paper_dir).as_posix(),
            ),
        },
        "split": {
            "seed": cfg.dataset.random_seed,
            "sizes": {
                "train": len(split.train_texts),
                "calibration": len(split.calib_texts),
                "test": len(split.test_texts),
            },
            "hashes": split_hashes,
        },
        "classical_predeclared_grid": classical,
        "lr_rejection_comparators": comparators,
        "distilbert_dense_grid_sensitivity": distil_dense,
        "llm_confidence_metrics": llm_confidence,
        "llm_fixed_break_even_policy": llm_fixed,
        "agreement_m9_reconstruction_matches_saved": m9_validation,
        "validations": {
            "all_primary_classical_rows_match_submitted_rounding": all(
                row["submitted_rounding_check"]["passes"] for row in classical
            ),
            "all_m9_agreement_scores_match_saved": all(m9_validation.values()),
            "no_llm_policy_threshold_selected_on_test": True,
            "llm_fixed_threshold_derivation": "route when 5*(1-p)<=1, hence p>=0.80",
        },
    }

    manifest_path = output_dir / "claim_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_csv(
        output_dir / "classical_operating_points.csv",
        [flatten_policy_record(row) for row in classical],
    )
    write_csv(
        output_dir / "lr_rejection_comparators.csv",
        [flatten_policy_record(row) for row in comparators],
    )
    write_csv(output_dir / "llm_confidence_metrics.csv", llm_confidence)
    write_csv(output_dir / "llm_fixed_break_even_policy.csv", llm_fixed)
    write_audit_report(output_dir / "AUDIT_REPORT.md", manifest)

    if not manifest["validations"]["all_primary_classical_rows_match_submitted_rounding"]:
        raise RuntimeError("a primary classical result no longer matches the submitted rounding")
    print(f"Wrote audited manifest and tables to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
