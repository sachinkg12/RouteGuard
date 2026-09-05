"""Select DistilBERT's abstention threshold on the CALIBRATION split (never test),
freeze it, and evaluate on the disjoint test split -- the leakage-free procedure
used for the other baselines. Reads the calib+test predictions saved by
scripts/distilbert_calibration_predict.py.

Run:
    .venv-arm64/bin/python scripts/distilbert_calibration_predict.py   # produces the JSON
    .venv/bin/python scripts/distilbert_calibration_select.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

C_OK, C_TRIAGE, C_WRONG = 0.0, 1.0, 5.0
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]
BOOT = 1000
SEED = 1337


def per_ticket_cost(correct, conf, tau):
    """Cost vector under threshold@tau at (0,1,5)."""
    out = np.empty(len(conf))
    for i, (c, p) in enumerate(zip(correct, conf)):
        if p >= tau:  # auto-route
            out[i] = C_OK if c else C_WRONG
        else:  # defer
            out[i] = C_TRIAGE
    return out


def main():
    d = json.loads(Path("outputs/distilbert_calib_test_preds.json").read_text())
    cal_ok = np.array([p == y for p, y in zip(d["calib_predictions"], d["calib_labels"])])
    cal_cf = np.array(d["calib_confidences"], float)
    te_ok = np.array([p == y for p, y in zip(d["test_predictions"], d["test_labels"])])
    te_cf = np.array(d["test_confidences"], float)

    acc_test = te_ok.mean()
    print(f"Retrained DistilBERT test accuracy = {acc_test:.4f} "
          f"(reference value 0.877; delta {acc_test-0.877:+.4f})")
    print(f"calib n={len(cal_ok)}  test n={len(te_ok)}")

    # --- select tau* on CALIBRATION ---
    cal_costs = {tau: per_ticket_cost(cal_ok, cal_cf, tau).mean() for tau in TAUS}
    tau_star = min(cal_costs, key=cal_costs.get)
    print("\ncalibration expected cost per tau:")
    for tau in TAUS:
        star = " <-- tau*" if tau == tau_star else ""
        print(f"  tau={tau:.1f}: {cal_costs[tau]:.4f}{star}")

    # --- evaluate frozen tau* on TEST ---
    always = np.where(te_ok, C_OK, C_WRONG)          # route everything
    thr = per_ticket_cost(te_ok, te_cf, tau_star)    # threshold@tau*
    e_always, e_thr = always.mean(), thr.mean()
    red = 1.0 - e_thr / e_always
    routed = te_cf >= tau_star
    cov = routed.mean()
    wrong_routed = (routed & ~te_ok).mean()          # wrong-route rate over all test
    on_routed_acc = te_ok[routed].mean() if routed.any() else float("nan")

    # --- paired bootstrap CI on the relative reduction ---
    rng = np.random.default_rng(SEED)
    n = len(te_ok)
    reds = []
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        a = always[idx].mean(); t = thr[idx].mean()
        reds.append(1.0 - t / a)
    lo, hi = np.percentile(reds, [2.5, 97.5])

    print("\n=== DistilBERT cost-baselines row (CALIBRATION-selected tau*) ===")
    print(f"  tau*                 = {tau_star:.2f}")
    print(f"  E[c] always-route    = {e_always:.4f}")
    print(f"  E[c] threshold@tau*  = {e_thr:.4f}")
    print(f"  reduction            = {100*red:.1f}%  95% CI [{100*lo:.1f}, {100*hi:.1f}]")
    print(f"  coverage @tau*       = {100*cov:.1f}%")
    print(f"  wrong-route rate     = {100*wrong_routed:.2f}%")
    print(f"  on-routed accuracy   = {100*on_routed_acc:.2f}%")
    print(f"\n  gap check: DistilBERT test acc {acc_test:.3f} vs best LLM 0.477 "
          f"= {acc_test-0.477:.3f}")


if __name__ == "__main__":
    main()
