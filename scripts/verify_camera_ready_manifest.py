"""Independently verify every numerical camera-ready claim from source inputs.

This checker intentionally does not import the manifest generator's helper
functions. It retrains the classical models, re-derives policies and intervals,
and re-analyses the frozen LLM/DistilBERT vectors with separate calculations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ticket_routing.data.loaders import build_loader_from_config
from ticket_routing.data.splitters import stratified_three_way_split
from ticket_routing.models.tfidf_logreg import TfidfLogisticRegressionPredictor
from ticket_routing.models.tfidf_rf import TfidfRandomForestPredictor
from ticket_routing.models.tfidf_svm import TfidfLinearSVMPredictor
from ticket_routing.utils.config import load_config
from ticket_routing.utils.hashing import sha256_iter_strings


CONFIG = REPO_ROOT / "configs" / "experiment_default.yaml"
LLM_RAW = (
    REPO_ROOT
    / "artifacts"
    / "ictai2026"
    / "llm_inputs"
)
TABLE_II_INPUT = REPO_ROOT / "artifacts" / "ictai2026" / "table_ii_common_subset.json"
DISTIL = REPO_ROOT / "outputs" / "distilbert_calib_test_preds.json"
BOOT = 1000
SEED = 1337


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual, expected, label: str, tolerance: float = 1e-12) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def policy_costs(correctness: Sequence[bool], scores: Sequence[float], tau: float):
    correct = np.asarray(correctness, dtype=bool)
    confidence = np.asarray(scores, dtype=float)
    routed = confidence >= tau
    policy = np.where(routed, np.where(correct, 0.0, 5.0), 1.0)
    always_route = np.where(correct, 0.0, 5.0)
    always_defer = np.ones(correct.size)
    return routed, policy, always_route, always_defer


def bootstrap_relative(policy: np.ndarray, baseline: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    samples = []
    n = policy.size
    for _ in range(BOOT):
        idx = rng.integers(0, n, size=n)
        denom = baseline[idx].mean()
        if denom != 0.0:
            samples.append(1.0 - policy[idx].mean() / denom)
    values = np.asarray(samples)
    return {
        "point_estimate": float(1.0 - policy.mean() / baseline.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n_bootstrap": BOOT,
        "n_finite_bootstrap": int(values.size),
        "seed": SEED,
    }


def bootstrap_difference(policy: np.ndarray, baseline: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    samples = np.empty(BOOT)
    n = policy.size
    for i in range(BOOT):
        idx = rng.integers(0, n, size=n)
        samples[i] = policy[idx].mean() - baseline[idx].mean()
    return {
        "point_estimate": float(policy.mean() - baseline.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_bootstrap": BOOT,
        "seed": SEED,
    }


def compare_interval(actual: dict, expected: dict, label: str, relative: bool) -> None:
    for key in ["point_estimate", "ci_low", "ci_high"]:
        close(actual[key], expected[key], f"{label}.{key}")
    equal(actual["n_bootstrap"], expected["n_bootstrap"], f"{label}.n_bootstrap")
    equal(actual["seed"], expected["seed"], f"{label}.seed")
    if relative:
        equal(
            actual["n_finite_bootstrap"],
            expected["n_finite_bootstrap"],
            f"{label}.n_finite_bootstrap",
        )


def compare_policy_record(
    record: dict,
    calibration_predictions: Sequence[str],
    calibration_labels: Sequence[str],
    calibration_scores: Sequence[float],
    test_predictions: Sequence[str],
    test_labels: Sequence[str],
    test_scores: Sequence[float],
    label: str,
) -> None:
    grid = [float(value) for value in record["selection"]["threshold_grid"]]
    calibration_correct = np.asarray(calibration_predictions) == np.asarray(calibration_labels)
    candidate_costs = {}
    for tau in grid:
        _, costs, _, _ = policy_costs(calibration_correct, calibration_scores, tau)
        candidate_costs[f"threshold@{tau:.2f}"] = float(costs.mean())
    selected_policy = min(candidate_costs, key=candidate_costs.get)
    selected_tau = float(selected_policy.split("@", 1)[1])
    equal(record["selection"]["selection_split"], "calibration", f"{label}.split")
    equal(record["selection"]["policy"], selected_policy, f"{label}.policy")
    close(record["selection"]["threshold"], selected_tau, f"{label}.tau")
    close(
        record["selection"]["expected_cost_per_ticket"],
        candidate_costs[selected_policy],
        f"{label}.calibration_cost",
    )
    for policy, value in candidate_costs.items():
        close(record["selection"]["candidate_costs"][policy], value, f"{label}.{policy}")

    correctness = np.asarray(test_predictions) == np.asarray(test_labels)
    routed, policy, always_route, always_defer = policy_costs(
        correctness, test_scores, selected_tau
    )
    test = record["test"]
    n_routed = int(routed.sum())
    expected_scalars = {
        "threshold": selected_tau,
        "n_test": int(correctness.size),
        "n_routed": n_routed,
        "n_deferred": int(correctness.size - n_routed),
        "coverage": float(routed.mean()),
        "accuracy_on_routed": float(correctness[routed].mean()) if n_routed else None,
        "wrong_route_rate": float(np.mean(routed & ~correctness)),
        "always_route_wrong_rate": float(np.mean(~correctness)),
        "always_route_cost": float(always_route.mean()),
        "always_defer_cost": 1.0,
        "policy_cost": float(policy.mean()),
    }
    for key, expected in expected_scalars.items():
        if isinstance(expected, int):
            equal(test[key], expected, f"{label}.test.{key}")
        else:
            close(test[key], expected, f"{label}.test.{key}")
    close(record["test_accuracy"], correctness.mean(), f"{label}.accuracy")
    close(
        record["test_macro_f1"],
        f1_score(
            test_labels,
            test_predictions,
            labels=sorted(set(test_labels)),
            average="macro",
            zero_division=0,
        ),
        f"{label}.macro_f1",
    )
    close(
        record["test_ece_10_equal_width_bins"],
        ece(np.asarray(test_scores, dtype=float), correctness),
        f"{label}.ece",
    )
    compare_interval(
        test["relative_reduction_vs_always_route"],
        bootstrap_relative(policy, always_route),
        f"{label}.relative_vs_route",
        True,
    )
    compare_interval(
        test["relative_reduction_vs_always_defer"],
        bootstrap_relative(policy, always_defer),
        f"{label}.relative_vs_defer",
        True,
    )
    compare_interval(
        test["cost_difference_vs_always_route"],
        bootstrap_difference(policy, always_route),
        f"{label}.difference_vs_route",
        False,
    )
    compare_interval(
        test["cost_difference_vs_always_defer"],
        bootstrap_difference(policy, always_defer),
        f"{label}.difference_vs_defer",
        False,
    )


def ece(scores: np.ndarray, correctness: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    total = 0.0
    for i in range(10):
        mask = (
            (scores >= edges[i]) & (scores <= edges[i + 1])
            if i == 9
            else (scores >= edges[i]) & (scores < edges[i + 1])
        )
        if mask.any():
            total += mask.mean() * abs(scores[mask].mean() - correctness[mask].mean())
    return float(total)


def tie_aurc(scores: np.ndarray, correctness: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    errors = 1 - correctness[order]
    area = 0.0
    admitted = 0
    cumulative_errors = 0
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_size = end - start
        admitted += group_size
        cumulative_errors += int(errors[start:end].sum())
        area += (group_size / scores.size) * (cumulative_errors / admitted)
        start = end
    return float(area)


def verify_inputs(manifest: dict, paper_dir: Path) -> tuple[int, list[str]]:
    checks = 0
    skipped: list[str] = []
    inputs = manifest["inputs"]
    simple = [
        "dataset",
        "default_config",
        "classical_config",
        "distilbert_predictions",
        "llm_common_labels",
    ]
    for key in simple:
        item = inputs[key]
        close_path = REPO_ROOT / item["path"]
        equal(file_hash(close_path), item["sha256"], f"input hash {key}")
        checks += 1
    for relative, expected_hash in inputs["llm_bundle_files"].items():
        equal(file_hash(REPO_ROOT / relative), expected_hash, f"input hash {relative}")
        checks += 1
    for key in ["submitted_source_zip", "submitted_supplement_zip"]:
        item = inputs[key]
        input_path = paper_dir / item["path"]
        if not input_path.exists():
            skipped.append(
                f"input hash {key}: private review-time archive is not redistributed"
            )
            continue
        equal(file_hash(input_path), item["sha256"], f"input hash {key}")
        checks += 1
    return checks, skipped


def verify_classical(manifest: dict, split) -> int:
    specs = {
        "TF-IDF + logistic regression": TfidfLogisticRegressionPredictor(
            max_features=100_000, ngram_max=2, C=1.0, random_state=42
        ),
        "TF-IDF + calibrated linear SVM": TfidfLinearSVMPredictor(
            max_features=100_000,
            ngram_max=2,
            C=1.0,
            random_state=42,
            calibration_cv=3,
        ),
        "TF-IDF + random forest": TfidfRandomForestPredictor(
            max_features=50_000,
            ngram_max=2,
            n_estimators=300,
            n_jobs=-1,
            random_state=42,
        ),
    }
    records = {row["model"]: row for row in manifest["classical_predeclared_grid"]}
    comparator_records = {
        row["confidence_score"]: row for row in manifest["lr_rejection_comparators"]
    }
    checks = 0
    for name, model in specs.items():
        print(f"Verifier fitting {name}...", flush=True)
        model.fit(split.train_texts, split.train_labels, split.label_names)
        calibration = model.predict(split.calib_texts)
        test = model.predict(split.test_texts)
        compare_policy_record(
            records[name],
            calibration.predicted_labels,
            split.calib_labels,
            calibration.confidence_scores or [],
            test.predicted_labels,
            split.test_labels,
            test.confidence_scores or [],
            name,
        )
        checks += 1
        if name == "TF-IDF + logistic regression":
            cal_proba = model.predict_proba_full(split.calib_texts)
            test_proba = model.predict_proba_full(split.test_texts)
            score_pairs = {
                "top_class_probability": (cal_proba.max(axis=1), test_proba.max(axis=1)),
                "top_two_probability_margin": (
                    margin(cal_proba),
                    margin(test_proba),
                ),
                "normalized_inverse_entropy": (
                    inverse_entropy(cal_proba),
                    inverse_entropy(test_proba),
                ),
            }
            for score_name, (cal_score, test_score) in score_pairs.items():
                compare_policy_record(
                    comparator_records[score_name],
                    calibration.predicted_labels,
                    split.calib_labels,
                    cal_score,
                    test.predicted_labels,
                    split.test_labels,
                    test_score,
                    f"LR comparator {score_name}",
                )
                checks += 1
    return checks


def margin(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def inverse_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    return 1.0 + np.sum(clipped * np.log(clipped), axis=1) / math.log(probabilities.shape[1])


def verify_distilbert(manifest: dict) -> int:
    data = json.loads(DISTIL.read_text())
    records = [
        manifest["classical_predeclared_grid"][-1],
        manifest["distilbert_dense_grid_sensitivity"],
    ]
    for index, record in enumerate(records):
        compare_policy_record(
            record,
            data["calib_predictions"],
            data["calib_labels"],
            data["calib_confidences"],
            data["test_predictions"],
            data["test_labels"],
            data["test_confidences"],
            f"DistilBERT analysis {index}",
        )
    return 2


def correctness(record: dict) -> np.ndarray:
    row = next(
        row
        for row in record["cost"]
        if row["policy"] == "always_route" and row["wrong_auto_route_cost"] == 5.0
    )
    return (np.asarray(row["per_ticket_costs"], dtype=float) == 0.0).astype(int)


def verify_llms(manifest: dict) -> int:
    raw_paths = sorted(LLM_RAW.glob("*__model_reported.json"))
    raw_paths = [path for path in raw_paths if not path.name.startswith("tfidf_logreg")]
    raw_records = {
        path.name.removesuffix("__model_reported.json"): json.loads(path.read_text())
        for path in raw_paths
    }
    predictions = {
        key: np.asarray(record["predictions"], dtype=object)
        for key, record in raw_records.items()
    }
    classical = json.loads((LLM_RAW / "tfidf_logreg__classical__model_reported.json").read_text())
    classical_predictions = np.asarray(classical["predictions"], dtype=object)
    common_subset = json.loads(TABLE_II_INPUT.read_text())
    true_labels = np.asarray(common_subset["true_labels"], dtype=object)
    projected_classical_predictions = np.asarray(
        common_subset["models"]["tfidf_logreg"]["predictions"], dtype=object
    )
    equal(true_labels.size, 1000, "LLM common-label count")
    equal(
        bool(np.array_equal(classical_predictions, projected_classical_predictions)),
        True,
        "LLM common-label alignment",
    )
    label_order = sorted(set(true_labels.tolist()))
    confidence_manifest = {
        (row["model_condition"], row["confidence_method"]): row
        for row in manifest["llm_confidence_metrics"]
    }
    fixed_manifest = {
        row["model_condition"]: row for row in manifest["llm_fixed_break_even_policy"]
    }

    checks = 0
    for key, raw in raw_records.items():
        prefix = key
        iso = json.loads((LLM_RAW / f"{prefix}__isotonic.json").read_text())
        agr = json.loads((LLM_RAW / f"{prefix}__agreement.json").read_text())
        outcome = correctness(raw)
        label_correctness = (
            np.asarray(raw["predictions"], dtype=object) == true_labels
        ).astype(int)
        equal(
            bool(np.array_equal(outcome, label_correctness)),
            True,
            f"{key}.label correctness",
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
        close(outcome.mean(), raw["classification"]["accuracy"], f"{key}.accuracy source")
        primary = predictions[key]
        llm_matches = np.vstack([candidate == primary for candidate in predictions.values()])
        m8 = llm_matches.mean(axis=0)
        m9 = (llm_matches.sum(axis=0) + (classical_predictions == primary)) / 9.0
        scores_by_method = {
            "raw_model_reported": np.asarray(raw["confidences"], dtype=float),
            "isotonic": np.asarray(iso["confidences"], dtype=float),
            "agreement_m9_mixed": m9,
            "agreement_m8_llm_only": m8,
        }
        equal(bool(np.allclose(m9, agr["confidences"], atol=1e-12)), True, f"{key}.M9")
        raw_ece = ece(scores_by_method["raw_model_reported"], outcome)
        raw_auroc = float(roc_auc_score(outcome, scores_by_method["raw_model_reported"]))
        for method, scores in scores_by_method.items():
            row = confidence_manifest[(key, method)]
            measured_ece = ece(scores, outcome)
            measured_auroc = float(roc_auc_score(outcome, scores))
            expected = {
                "n_test": 1000,
                "accuracy": float(outcome.mean()),
                "macro_f1": macro_f1,
                "ece_10_equal_width_bins": measured_ece,
                "relative_ece_reduction_vs_raw": 1.0 - measured_ece / raw_ece,
                "correctness_auroc": measured_auroc,
                "correctness_auroc_delta_vs_raw": measured_auroc - raw_auroc,
                "tie_group_aurc": tie_aurc(scores, outcome),
                "n_unique_scores": int(np.unique(scores).size),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
            }
            for field, value in expected.items():
                if isinstance(value, int):
                    equal(row[field], value, f"{key}.{method}.{field}")
                else:
                    close(row[field], value, f"{key}.{method}.{field}")
            if method in {"raw_model_reported", "isotonic", "agreement_m9_mixed"}:
                source_record = {
                    "raw_model_reported": raw,
                    "isotonic": iso,
                    "agreement_m9_mixed": agr,
                }[method]
                close(
                    measured_ece,
                    source_record["calibration"]["ece"],
                    f"{key}.{method}.saved_ece",
                )
            checks += 1

        fixed = fixed_manifest[key]
        routed, policy, always_route, always_defer = policy_costs(
            outcome, scores_by_method["isotonic"], 0.8
        )
        expected_scalars = {
            "threshold": 0.8,
            "n_test": 1000,
            "n_routed": int(routed.sum()),
            "n_deferred": int(1000 - routed.sum()),
            "coverage": float(routed.mean()),
            "accuracy_on_routed": float(outcome[routed].mean()) if routed.any() else None,
            "wrong_route_rate": float(np.mean(routed & ~outcome.astype(bool))),
            "always_route_wrong_rate": float(np.mean(~outcome.astype(bool))),
            "always_route_cost": float(always_route.mean()),
            "always_defer_cost": 1.0,
            "policy_cost": float(policy.mean()),
        }
        for field, value in expected_scalars.items():
            if isinstance(value, int):
                equal(fixed[field], value, f"{key}.fixed.{field}")
            else:
                close(fixed[field], value, f"{key}.fixed.{field}")
        compare_interval(
            fixed["relative_reduction_vs_always_route"],
            bootstrap_relative(policy, always_route),
            f"{key}.fixed.relative_vs_route",
            True,
        )
        compare_interval(
            fixed["relative_reduction_vs_always_defer"],
            bootstrap_relative(policy, always_defer),
            f"{key}.fixed.relative_vs_defer",
            True,
        )
        compare_interval(
            fixed["cost_difference_vs_always_route"],
            bootstrap_difference(policy, always_route),
            f"{key}.fixed.difference_vs_route",
            False,
        )
        compare_interval(
            fixed["cost_difference_vs_always_defer"],
            bootstrap_difference(policy, always_defer),
            f"{key}.fixed.difference_vs_defer",
            False,
        )
        checks += 1
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", required=True, type=Path)
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=None,
        help=(
            "Optional path for a newly generated verification receipt. By default "
            "the verifier is read-only so checking an extracted release does not "
            "invalidate its packaged checksums."
        ),
    )
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    manifest_path = paper_dir / "camera_ready_results" / "claim_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    checks, skipped = verify_inputs(manifest, paper_dir)
    cfg = load_config(CONFIG)
    bundle = build_loader_from_config(cfg.dataset).load()
    split = stratified_three_way_split(
        bundle,
        train_fraction=cfg.split.train_fraction,
        calibration_fraction=cfg.split.calibration_fraction,
        test_fraction=cfg.split.test_fraction,
        seed=cfg.dataset.random_seed,
    )
    equal(len(bundle), manifest["scope"]["n_examples"], "dataset size")
    expected_hashes = {
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
    equal(expected_hashes, manifest["split"]["hashes"], "split hashes")
    checks += 4
    checks += verify_classical(manifest, split)
    checks += verify_distilbert(manifest)
    checks += verify_llms(manifest)

    receipt = {
        "status": "PASS",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": file_hash(manifest_path),
        "verifier": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "verifier_sha256": file_hash(Path(__file__).resolve()),
        "independent_check_count": checks,
        "statement": (
            "All numerical fields present in claim_manifest.json—including its listed "
            "source and split hashes, classical operating points and bootstrap intervals, "
            "LLM confidence and agreement metrics, rejection comparators, and dense "
            "DistilBERT sensitivity—were independently recomputed from the manifest's "
            "listed inputs. Secondary analyses and manuscript values outside the manifest "
            "are not covered by this receipt."
        ),
    }
    if args.receipt_output is not None:
        if skipped:
            raise RuntimeError(
                "Refusing to issue a release receipt with unavailable inputs: "
                + "; ".join(skipped)
            )
        receipt_path = args.receipt_output.resolve()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"PASS: {checks} independent checks; receipt: {receipt_path}", flush=True)
    else:
        if skipped:
            for item in skipped:
                print(f"SKIP: {item}", flush=True)
            print(
                f"PASS_WITH_SKIPS: {checks} independent checks; "
                f"{len(skipped)} private historical hash checks skipped; "
                "no files modified",
                flush=True,
            )
        else:
            print(f"PASS: {checks} independent checks; no files modified", flush=True)


if __name__ == "__main__":
    main()
