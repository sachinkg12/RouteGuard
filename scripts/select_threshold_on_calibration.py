"""Threshold selection on the calibration split, evaluation on test.

Selects the operating threshold on a held-out calibration split rather than on the
test set, so threshold@0.60 in the headline cost analysis is not chosen in hindsight.
This script:

1. Loads the public IT-ticket dataset and reproduces the stratified
   70/10/20 split with seed 42 (same as the headline runs).
2. Trains the TF-IDF + LR classifier on the train split.
3. Predicts on the calibration split, gets per-ticket confidences.
4. For each threshold tau in {0.5, 0.6, 0.7, 0.8, 0.9}, computes the expected
   cost per ticket on the CALIBRATION split under the canonical (0, 1, 5) cost.
5. Picks the calibration-best threshold tau*.
6. Reports the test-set expected cost at tau* (using held-out test predictions).

If calibration-best == 0.60 (the current headline), the headline is unchanged
and we just add the procedure sentence. If it differs, we report the
calibration-selected number.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from ticket_routing.data.loaders import build_loader_from_config
from ticket_routing.data.splitters import stratified_three_way_split
from ticket_routing.models.tfidf_logreg import TfidfLogisticRegressionPredictor
from ticket_routing.utils.config import load_config


def cost_at_threshold(
    confidences, predicted_labels, true_labels, tau, c_ok=0.0, c_triage=1.0, c_wrong=5.0
):
    """Expected cost per ticket under threshold@tau."""
    n = len(confidences)
    total = 0.0
    per_ticket = []
    for c, pred, true in zip(confidences, predicted_labels, true_labels):
        if c < tau:
            cost = c_triage
        elif pred == true:
            cost = c_ok
        else:
            cost = c_wrong
        per_ticket.append(cost)
        total += cost
    return total / n, per_ticket


def coverage_acc(confidences, predicted_labels, true_labels, tau):
    n = len(confidences)
    routed = sum(1 for c in confidences if c >= tau)
    correct_routed = sum(
        1 for c, p, t in zip(confidences, predicted_labels, true_labels)
        if c >= tau and p == t
    )
    cov = routed / n
    acc = correct_routed / routed if routed else 0.0
    return cov, acc


def main():
    cfg = load_config("configs/experiment_default.yaml")
    loader = build_loader_from_config(cfg.dataset)
    bundle = loader.load()
    split = stratified_three_way_split(
        bundle,
        train_fraction=cfg.split.train_fraction,
        calibration_fraction=cfg.split.calibration_fraction,
        test_fraction=cfg.split.test_fraction,
        seed=cfg.dataset.random_seed,
    )
    print(f"Split sizes: train={len(split.train_texts)}, "
          f"calib={len(split.calib_texts)}, test={len(split.test_texts)}")

    clf = TfidfLogisticRegressionPredictor(
        max_features=cfg.classical.max_features,
        ngram_max=cfg.classical.ngram_max,
        C=cfg.classical.C,
        random_state=cfg.dataset.random_seed,
    )
    clf.fit(split.train_texts, split.train_labels, split.label_names)

    calib_batch = clf.predict(split.calib_texts)
    test_batch = clf.predict(split.test_texts)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    print("\n=== CALIBRATION SPLIT (n={}) ===".format(len(split.calib_texts)))
    print(f"{'tau':>5}  {'E[cost]':>10}  {'coverage':>10}  {'acc_routed':>11}")
    calib_costs = {}
    for tau in thresholds:
        cost, _ = cost_at_threshold(
            calib_batch.confidence_scores,
            calib_batch.predicted_labels,
            split.calib_labels,
            tau,
        )
        cov, acc = coverage_acc(
            calib_batch.confidence_scores,
            calib_batch.predicted_labels,
            split.calib_labels,
            tau,
        )
        calib_costs[tau] = cost
        print(f"{tau:>5.2f}  {cost:>10.4f}  {cov*100:>9.1f}%  {acc*100:>10.1f}%")

    # Always-route baseline on calibration
    always_calib, _ = cost_at_threshold(
        calib_batch.confidence_scores,
        calib_batch.predicted_labels,
        split.calib_labels,
        tau=0.0,
    )
    print(f"\n  always_route on calibration: E[cost]={always_calib:.4f}")

    tau_star = min(calib_costs, key=calib_costs.get)
    print(f"\nCALIBRATION-BEST threshold: tau*={tau_star:.2f} "
          f"(cost on calib = {calib_costs[tau_star]:.4f})")

    print("\n=== TEST SPLIT (n={}) ===".format(len(split.test_texts)))
    print(f"{'tau':>5}  {'E[cost]':>10}  {'coverage':>10}  {'acc_routed':>11}")
    test_costs = {}
    for tau in thresholds:
        cost, per = cost_at_threshold(
            test_batch.confidence_scores,
            test_batch.predicted_labels,
            split.test_labels,
            tau,
        )
        cov, acc = coverage_acc(
            test_batch.confidence_scores,
            test_batch.predicted_labels,
            split.test_labels,
            tau,
        )
        test_costs[tau] = (cost, per)
        print(f"{tau:>5.2f}  {cost:>10.4f}  {cov*100:>9.1f}%  {acc*100:>10.1f}%")

    always_test_cost, always_test_per = cost_at_threshold(
        test_batch.confidence_scores,
        test_batch.predicted_labels,
        split.test_labels,
        tau=0.0,
    )
    print(f"\n  always_route on test: E[cost]={always_test_cost:.4f}")

    cost_at_star, per_at_star = test_costs[tau_star]
    rel_reduction = (always_test_cost - cost_at_star) / always_test_cost
    print("\n=== HEADLINE (calibration-selected, test-evaluated) ===")
    print(f"Selected on calibration: tau* = {tau_star:.2f}")
    print(f"Evaluated on test:")
    print(f"  always_route   E[cost] = {always_test_cost:.4f}")
    print(f"  threshold@{tau_star:.2f}  E[cost] = {cost_at_star:.4f}")
    print(f"  relative reduction = {rel_reduction*100:.2f}%")

    # Bootstrap CI (1000 paired resamples, seed 1337)
    rng = np.random.default_rng(1337)
    diffs = []
    per_always = np.array(always_test_per)
    per_star = np.array(per_at_star)
    n = len(per_always)
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        d = per_star[idx].mean() - per_always[idx].mean()
        diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  bootstrap 95% CI on cost delta: [{lo:.4f}, {hi:.4f}]")
    rel_lo = -hi / always_test_cost
    rel_hi = -lo / always_test_cost
    print(f"  bootstrap 95% CI on relative reduction: "
          f"[{rel_lo*100:.2f}%, {rel_hi*100:.2f}%]")


if __name__ == "__main__":
    main()
