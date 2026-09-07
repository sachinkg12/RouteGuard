"""Recompute Table II on the common 1,000-ticket test subset.

Inputs are a sanitized camera-ready artifact containing only the common-subset
labels, predictions, confidences, and review-time source hashes. No ticket text
or nested historical archive is required.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.metrics import f1_score

from ticket_routing.evaluation.calibration import expected_calibration_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/ictai2026/table_ii_common_subset.json"),
    )
    args = parser.parse_args()
    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    sub_labels = artifact["true_labels"]
    models = artifact["models"]
    expected_n = artifact["provenance"]["subsample_n"]
    if len(sub_labels) != expected_n:
        raise ValueError("true-label length does not match artifact metadata")
    print(f"Loaded {expected_n} sanitized common-subset rows")

    print(f"\n{'Baseline':<24} {'Acc':>6} {'Macro-F1':>9} {'ECE-raw':>8}")
    print("-" * 50)
    for name, data in models.items():
        preds_sub = data["predictions"]
        confs_sub = data["confidences"]
        if len(preds_sub) != expected_n or len(confs_sub) != expected_n:
            raise ValueError(f"{name}: vector length does not match artifact metadata")
        # Compute metrics
        correct = sum(1 for p, t in zip(preds_sub, sub_labels) if p == t)
        acc = correct / len(sub_labels)
        macro = f1_score(sub_labels, preds_sub, average="macro", zero_division=0)
        correctness = [1 if p == t else 0 for p, t in zip(preds_sub, sub_labels)]
        ece = expected_calibration_error(confs_sub, correctness, n_bins=10)["ece"]
        print(f"{name:<24} {acc:>6.3f} {macro:>9.3f} {ece:>8.3f}")


if __name__ == "__main__":
    main()
