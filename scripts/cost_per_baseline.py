"""Per-baseline calibration-selected threshold and test cost.

Refits SVM and RF on train, selects each model's tau* on calibration,
evaluates on the held-out test split. For DistilBERT we cannot refit
cheaply, so we evaluate at the shared substrate threshold tau=0.60
(the classical baseline's calibration-selected value).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from ticket_routing.data.loaders import build_loader_from_config
from ticket_routing.data.splitters import stratified_three_way_split
from ticket_routing.models.tfidf_logreg import TfidfLogisticRegressionPredictor
from ticket_routing.models.tfidf_svm import TfidfLinearSVMPredictor
from ticket_routing.models.tfidf_rf import TfidfRandomForestPredictor
from ticket_routing.utils.config import load_config


def cost_at_tau(confs, preds, trues, tau, c_ok=0.0, c_triage=1.0, c_wrong=5.0):
    n = len(confs)
    total = 0.0
    per = []
    for c, p, t in zip(confs, preds, trues):
        if c < tau:
            cost = c_triage
        elif p == t:
            cost = c_ok
        else:
            cost = c_wrong
        per.append(cost)
        total += cost
    return total / n, np.array(per)


def coverage_acc(confs, preds, trues, tau):
    n = len(confs)
    routed_correct = routed = 0
    for c, p, t in zip(confs, preds, trues):
        if c >= tau:
            routed += 1
            if p == t:
                routed_correct += 1
    cov = routed / n
    acc = routed_correct / routed if routed else 0.0
    wrong_rate = (routed - routed_correct) / n
    return cov, acc, wrong_rate


def evaluate_model(name, clf, split):
    print(f"\n=== {name} ===")
    clf.fit(split.train_texts, split.train_labels, split.label_names)
    calib_batch = clf.predict(split.calib_texts)
    test_batch = clf.predict(split.test_texts)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    calib_costs = {}
    for tau in thresholds:
        c, _ = cost_at_tau(
            calib_batch.confidence_scores, calib_batch.predicted_labels,
            split.calib_labels, tau,
        )
        calib_costs[tau] = c

    tau_star = min(calib_costs, key=calib_costs.get)
    always_test, always_per = cost_at_tau(
        test_batch.confidence_scores, test_batch.predicted_labels,
        split.test_labels, tau=0.0,
    )
    cost_star, per_star = cost_at_tau(
        test_batch.confidence_scores, test_batch.predicted_labels,
        split.test_labels, tau=tau_star,
    )
    cov, acc, wrong_rate = coverage_acc(
        test_batch.confidence_scores, test_batch.predicted_labels,
        split.test_labels, tau_star,
    )
    rel = (always_test - cost_star) / always_test if always_test else 0.0

    # Bootstrap CI on relative reduction
    rng = np.random.default_rng(1337)
    diffs = []
    n = len(always_per)
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        diffs.append(per_star[idx].mean() - always_per[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rel_lo = (-hi / always_test) if always_test else 0.0
    rel_hi = (-lo / always_test) if always_test else 0.0

    print(f"  Calibration U-curve: " +
          " ".join(f"{t}={calib_costs[t]:.3f}" for t in thresholds))
    print(f"  tau* (calibration-selected) = {tau_star:.2f}")
    print(f"  always_route on test   E[cost] = {always_test:.4f}")
    print(f"  threshold@{tau_star:.2f} on test E[cost] = {cost_star:.4f}")
    print(f"  relative reduction = {rel*100:.2f}%")
    print(f"  bootstrap CI = [{rel_lo*100:.2f}%, {rel_hi*100:.2f}%]")
    print(f"  coverage = {cov*100:.1f}%, acc on routed = {acc*100:.1f}%, "
          f"wrong-route rate = {wrong_rate*100:.2f}%")
    return {
        "name": name, "tau_star": tau_star,
        "always_cost": always_test, "tau_star_cost": cost_star,
        "rel": rel, "rel_ci": (rel_lo, rel_hi),
        "coverage": cov, "acc_routed": acc, "wrong_rate": wrong_rate,
    }


def evaluate_distilbert_at_shared_tau(shared_tau, split):
    """DistilBERT predictions are loaded from disk (no calibration available)."""
    print(f"\n=== DistilBERT (eval at shared tau={shared_tau}, "
          f"no separate calibration available) ===")
    raw_path = Path("outputs/paper_distilbert__20260521_230540/paper_bundle/"
                    "raw/distilbert_finetuned__classical__model_reported.json")
    data = json.loads(raw_path.read_text())
    always_test, always_per = cost_at_tau(
        data["confidences"], data["predictions"], split.test_labels, tau=0.0,
    )
    cost_t, per_t = cost_at_tau(
        data["confidences"], data["predictions"], split.test_labels, tau=shared_tau,
    )
    cov, acc, wrong_rate = coverage_acc(
        data["confidences"], data["predictions"], split.test_labels, shared_tau,
    )
    rel = (always_test - cost_t) / always_test

    rng = np.random.default_rng(1337)
    diffs = []
    n = len(always_per)
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        diffs.append(per_t[idx].mean() - always_per[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rel_lo, rel_hi = -hi / always_test, -lo / always_test

    print(f"  always_route on test E[cost] = {always_test:.4f}")
    print(f"  threshold@{shared_tau} on test E[cost] = {cost_t:.4f}")
    print(f"  relative reduction = {rel*100:.2f}%")
    print(f"  bootstrap CI = [{rel_lo*100:.2f}%, {rel_hi*100:.2f}%]")
    print(f"  coverage = {cov*100:.1f}%, acc on routed = {acc*100:.1f}%, "
          f"wrong-route rate = {wrong_rate*100:.2f}%")


def main():
    cfg = load_config("configs/experiment_default.yaml")
    bundle = build_loader_from_config(cfg.dataset).load()
    split = stratified_three_way_split(
        bundle,
        train_fraction=cfg.split.train_fraction,
        calibration_fraction=cfg.split.calibration_fraction,
        test_fraction=cfg.split.test_fraction,
        seed=cfg.dataset.random_seed,
    )

    lr = TfidfLogisticRegressionPredictor(
        max_features=cfg.classical.max_features, ngram_max=cfg.classical.ngram_max,
        C=cfg.classical.C, random_state=cfg.dataset.random_seed,
    )
    svm = TfidfLinearSVMPredictor(
        max_features=100000, ngram_max=2, C=1.0,
        random_state=cfg.dataset.random_seed, calibration_cv=3,
    )
    rf = TfidfRandomForestPredictor(
        max_features=50000, ngram_max=2, n_estimators=300, n_jobs=-1,
        random_state=cfg.dataset.random_seed,
    )

    evaluate_model("TF-IDF + LR", lr, split)
    evaluate_model("TF-IDF + Linear SVM", svm, split)
    evaluate_model("TF-IDF + Random Forest", rf, split)
    evaluate_distilbert_at_shared_tau(0.60, split)


if __name__ == "__main__":
    main()
