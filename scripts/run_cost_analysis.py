"""Re-runs the cost-sensitivity sweep over a saved run's raw JSON results.

This is useful when you want to add new wrong-route cost options without
re-running predictors.

Usage:
    python scripts/run_cost_analysis.py \
        --run-dir outputs/debug_synth__YYYYMMDD_HHMMSS \
        --wrong-route-costs 1 3 5 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ticket_routing.abstention.always_route import AlwaysRoutePolicy
from ticket_routing.abstention.threshold_policy import ThresholdAbstentionPolicy
from ticket_routing.evaluation.cost import CostModel, cost_table_row, per_ticket_costs
from ticket_routing.models.base import PredictionBatch
from ticket_routing.utils.logging import get_logger


def _batch_from_raw(raw: dict) -> PredictionBatch:
    return PredictionBatch(
        predicted_labels=raw["predictions"],
        confidence_scores=raw["confidences"],
        raw_outputs=None,
        parse_status=raw["parse_status"] or None,
        model_metadata=raw.get("metadata", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--wrong-route-costs", nargs="+", type=float, default=[2.0, 5.0, 10.0]
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9]
    )
    parser.add_argument("--human-triage", type=float, default=1.0)
    parser.add_argument("--correct-auto", type=float, default=0.0)
    args = parser.parse_args()

    logger = get_logger("run_cost_analysis")
    run_dir = Path(args.run_dir)
    raw_dir = run_dir / "paper_bundle" / "raw"
    if not raw_dir.exists():
        raise SystemExit(f"No raw results at {raw_dir}")

    out_rows = []
    for raw_file in sorted(raw_dir.glob("*.json")):
        raw = json.loads(raw_file.read_text())
        batch = _batch_from_raw(raw)
        # Recover per-ticket correctness from the saved always_route cost row:
        # under always-route a ticket costs 0 iff it was routed correctly and
        # wrong_route_cost (>0) otherwise (c_ok = 0). Correctness is therefore
        # decodable from the serialized cost rows without needing the raw labels.
        always_row = next(
            (c for c in raw.get("cost", []) if c.get("policy") == "always_route"),
            None,
        )
        if not always_row or not always_row.get("per_ticket_costs"):
            logger.warning("No always_route per-ticket costs in %s; skipping.", raw_file.name)
            continue
        correct = [1 if c == 0 else 0 for c in always_row["per_ticket_costs"]]
        # Re-derive decisions for each threshold using saved confidences.
        for wcost in args.wrong_route_costs:
            cm = CostModel(
                correct_auto_route=args.correct_auto,
                human_triage=args.human_triage,
                wrong_auto_route=wcost,
            )
            for thresh in args.thresholds:
                pol = ThresholdAbstentionPolicy(threshold=thresh)
                decisions = pol.decide(batch, batch.confidence_scores or [])
                # Simulated per-ticket cost using correctness signal we derived.
                sim_per_ticket = []
                for d, c in zip(decisions, correct):
                    if d == "DEFER_TO_HUMAN":
                        sim_per_ticket.append(cm.human_triage)
                    elif c == 1:
                        sim_per_ticket.append(cm.correct_auto_route)
                    else:
                        sim_per_ticket.append(cm.wrong_auto_route)
                out_rows.append(
                    {
                        "source": raw_file.name,
                        "policy": pol.name,
                        "wrong_route_cost": wcost,
                        "expected_cost_per_ticket": sum(sim_per_ticket) / max(1, len(sim_per_ticket)),
                    }
                )

    out_path = run_dir / "paper_bundle" / "tables" / "extra_cost_sensitivity.csv"
    if out_rows:
        import pandas as pd

        pd.DataFrame(out_rows).to_csv(out_path, index=False)
        logger.info("Wrote %s (%d rows).", out_path, len(out_rows))
    else:
        logger.warning("No rows to write.")


if __name__ == "__main__":
    main()
