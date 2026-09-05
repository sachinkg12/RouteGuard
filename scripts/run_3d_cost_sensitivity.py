"""3D cost-model sensitivity sweep over a saved run's raw JSON.

Reconstructs each ticket's (decision, was_correct) from the saved per-policy
per-ticket cost vectors, then sweeps a Cartesian grid over
(correct_auto_route, human_triage, wrong_auto_route) without any re-prediction
or API calls. Writes a tidy CSV ready for plotting.

Usage:
    python scripts/run_3d_cost_sensitivity.py \
        --run-dir outputs/paper_llm_4way__20260521_154455 \
        --correct-auto 0 \
        --human-triage 0.5 1.0 1.5 \
        --wrong-auto 2 5 10 20

A new CSV lands at <run-dir>/paper_bundle/tables/extra_cost_3d_sweep.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from ticket_routing.utils.logging import get_logger


def _decode_decisions_and_correctness(cost_rows):
    """Recover per-ticket (decision_kind, was_correct) from saved per_ticket_costs.

    Each ticket gets one of three categories given the canonical (0, h, w) cost model:
      cost == correct_auto_route  -> AUTO_ROUTE & correct
      cost == wrong_auto_route    -> AUTO_ROUTE & wrong
      cost == human_triage        -> DEFER
    We pick a row with distinct (0, h, w) values to decode.
    """
    # Find a row whose three configured costs are distinct.
    for c in cost_rows:
        per_t = c.get("per_ticket_costs") or []
        if not per_t:
            continue
        correct_v = c["correct_auto_route_cost"]
        wrong_v = c["wrong_auto_route_cost"]
        defer_v = c["human_triage_cost"]
        if len({correct_v, wrong_v, defer_v}) != 3:
            continue
        out = []
        for v in per_t:
            if v == correct_v:
                out.append(("AUTO_ROUTE", True))
            elif v == wrong_v:
                out.append(("AUTO_ROUTE", False))
            elif v == defer_v:
                out.append(("DEFER", None))
            else:
                # Should not happen under the canonical model.
                out.append(("UNKNOWN", None))
        return c["policy"], out
    return None, None


def _resimulate(decisions, correctness, correct_auto, human_triage, wrong_auto):
    total = 0.0
    n = 0
    for d, c in zip(decisions, correctness):
        n += 1
        if d == "DEFER":
            total += human_triage
        elif d == "AUTO_ROUTE" and c is True:
            total += correct_auto
        elif d == "AUTO_ROUTE" and c is False:
            total += wrong_auto
        else:
            n -= 1
    return total / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--correct-auto", nargs="+", type=float, default=[0.0])
    parser.add_argument("--human-triage", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--wrong-auto", nargs="+", type=float, default=[2.0, 5.0, 10.0, 20.0])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["always_route", "threshold@0.50", "threshold@0.60", "threshold@0.70",
                 "threshold@0.80", "threshold@0.90"],
    )
    args = parser.parse_args()

    logger = get_logger("3d_cost_sensitivity")
    run_dir = Path(args.run_dir)
    raw_dir = run_dir / "paper_bundle" / "raw"
    if not raw_dir.exists():
        raise SystemExit(f"No raw results at {raw_dir}")

    rows = []
    for raw_file in sorted(raw_dir.glob("*.json")):
        raw = json.loads(raw_file.read_text())
        cost_rows = raw.get("cost", [])
        # For each policy of interest, decode decisions+correctness once.
        for policy in args.policies:
            pol_rows = [c for c in cost_rows if c["policy"] == policy]
            if not pol_rows:
                continue
            policy_name, decoded = _decode_decisions_and_correctness(pol_rows)
            if not decoded:
                continue
            decisions = [d for d, _ in decoded]
            correctness = [c for _, c in decoded]
            n_auto = sum(1 for d in decisions if d == "AUTO_ROUTE")
            n_defer = sum(1 for d in decisions if d == "DEFER")
            for ca, ht, wa in itertools.product(args.correct_auto, args.human_triage, args.wrong_auto):
                ecost = _resimulate(decisions, correctness, ca, ht, wa)
                rows.append(
                    {
                        "source": raw_file.stem,
                        "predictor_name": raw.get("predictor_name"),
                        "setting": raw.get("setting"),
                        "policy": policy,
                        "correct_auto_route": ca,
                        "human_triage": ht,
                        "wrong_auto_route": wa,
                        "expected_cost_per_ticket": ecost,
                        "coverage": n_auto / max(1, len(decisions)),
                        "deferred_rate": n_defer / max(1, len(decisions)),
                    }
                )

    out_path = run_dir / "paper_bundle" / "tables" / "extra_cost_3d_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows).", out_path, len(rows))
        # Compact summary: best policy per (predictor, ht, wa) combination.
        idx = df.groupby(["predictor_name", "setting", "human_triage", "wrong_auto_route"])[
            "expected_cost_per_ticket"
        ].idxmin()
        best = df.loc[idx, ["predictor_name", "setting", "human_triage", "wrong_auto_route",
                            "policy", "expected_cost_per_ticket", "coverage"]]
        summary_path = out_path.with_name("extra_cost_3d_summary.csv")
        best.to_csv(summary_path, index=False)
        logger.info("Wrote %s (%d best-policy rows).", summary_path, len(best))
    else:
        logger.warning("No rows produced — empty raw dir?")


if __name__ == "__main__":
    main()
