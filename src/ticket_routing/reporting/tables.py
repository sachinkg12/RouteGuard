"""Generates the six result tables as markdown + CSV."""
from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from ..data.schema import SplitBundle


# ---------- Table 1 ----------------------------------------------------------

def dataset_summary_table(bundle, split: SplitBundle) -> pd.DataFrame:
    label_counts = Counter(bundle.labels)
    if label_counts:
        max_c = max(label_counts.values())
        min_c = min(label_counts.values())
        imbalance = f"max/min = {max_c}/{min_c} ({max_c / max(1, min_c):.2f}x)"
    else:
        imbalance = "n/a"
    median_len = (
        int(statistics.median(len(t) for t in bundle.texts)) if bundle.texts else 0
    )
    return pd.DataFrame(
        [
            {
                "n_tickets": len(bundle.texts),
                "n_classes": len(bundle.label_names),
                "train_size": len(split.train_texts),
                "calibration_size": len(split.calib_texts),
                "test_size": len(split.test_texts),
                "median_ticket_length": median_len,
                "class_imbalance": imbalance,
            }
        ]
    )


# ---------- Table 2 ----------------------------------------------------------

def classification_table(results) -> pd.DataFrame:
    rows = []
    seen = set()
    for r in results:
        # Classification metrics are confidence-method-independent; emit one row
        # per (model, setting) instead of a duplicate per confidence method.
        key = (r.predictor_name, r.setting)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "model": r.predictor_name,
                "setting": r.setting,
                "accuracy": r.classification["accuracy"],
                "macro_f1": r.classification["macro_f1"],
                "weighted_f1": r.classification["weighted_f1"],
                "parse_error_rate": r.classification["parse_error_rate"],
                "invalid_label_rate": r.classification["invalid_label_rate"],
            }
        )
    return pd.DataFrame(rows)


# ---------- Table 3 ----------------------------------------------------------

def calibration_table(results) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.predictor_name,
                "setting": r.setting,
                "confidence_method": r.confidence_method,
                "ece": r.calibration["ece"],
                "mean_confidence": r.calibration["mean_confidence"],
                "accuracy": r.calibration["accuracy"],
            }
        )
    return pd.DataFrame(rows)


# ---------- Table 4 ----------------------------------------------------------

def abstention_table(results) -> pd.DataFrame:
    rows = []
    for r in results:
        for a in r.abstention:
            # Parse threshold out of policy name when present.
            threshold = None
            if a["policy"].startswith("threshold@"):
                threshold = float(a["policy"].split("@", 1)[1])
            rows.append(
                {
                    "model": r.predictor_name,
                    "setting": r.setting,
                    "confidence_method": r.confidence_method,
                    "policy": a["policy"],
                    "threshold": threshold,
                    "coverage": a["coverage"],
                    "deferred_rate": a["deferred_rate"],
                    "accuracy_on_auto_routed": a["accuracy_on_auto_routed"],
                    "wrong_auto_route_rate": a["wrong_auto_route_rate"],
                }
            )
    return pd.DataFrame(rows)


# ---------- Table 5 ----------------------------------------------------------

def cost_table(results) -> pd.DataFrame:
    rows = []
    for r in results:
        ci = (r.metadata or {}).get("bootstrap", {}).get(
            "best_threshold_vs_always_route_cost_per_ticket"
        )
        ci_dict = ci if isinstance(ci, dict) else {}
        # The bootstrap CI was computed for exactly ONE row: the cost-minimizing
        # threshold policy at the default wrong-route cost. Attach it only there,
        # never broadcast it across every threshold row.
        best_policy = ci_dict.get("best_policy")
        best_wrong_cost = ci_dict.get("wrong_route_cost")
        for c in r.cost:
            attach_ci = (
                best_policy is not None
                and c["policy"] == best_policy
                and c["wrong_auto_route_cost"] == best_wrong_cost
            )
            rows.append(
                {
                    "model": r.predictor_name,
                    "setting": r.setting,
                    "confidence_method": r.confidence_method,
                    "policy": c["policy"],
                    "wrong_route_cost": c["wrong_auto_route_cost"],
                    "expected_cost_per_ticket": c["expected_cost_per_ticket"],
                    "total_cost": c["total_cost"],
                    "cost_reduction_vs_always_route": c["cost_reduction_vs_always_route"],
                    "cost_reduction_vs_always_defer": c["cost_reduction_vs_always_defer"],
                    "cost_delta_vs_always_route_ci_low": ci_dict.get("ci_low") if attach_ci else None,
                    "cost_delta_vs_always_route_ci_high": ci_dict.get("ci_high") if attach_ci else None,
                }
            )
    return pd.DataFrame(rows)


# ---------- Table 6 ----------------------------------------------------------

def error_analysis_table(results) -> pd.DataFrame:
    rows = []
    seen = set()
    for r in results:
        # Confusions are confidence-method-independent; one block per (model, setting).
        key = (r.predictor_name, r.setting)
        if key in seen:
            continue
        seen.add(key)
        for pair in r.error_analysis:
            rows.append(
                {
                    "model": r.predictor_name,
                    "setting": r.setting,
                    "true_label": pair["true_label"],
                    "predicted_label": pair["predicted_label"],
                    "count": pair["count"],
                    "example_ticket": pair["example_ticket"][:200],
                }
            )
    return pd.DataFrame(rows)


# ---------- IO ---------------------------------------------------------------

def save_tables(out_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


def tables_as_markdown(tables: dict[str, pd.DataFrame]) -> str:
    blocks: List[str] = []
    for name, df in tables.items():
        blocks.append(f"### {name}\n")
        blocks.append(df.to_markdown(index=False, floatfmt=".4f"))
        blocks.append("\n")
    return "\n".join(blocks)
