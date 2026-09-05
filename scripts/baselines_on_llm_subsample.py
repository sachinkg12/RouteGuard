"""Evaluate all non-LLM baselines on the same 1000-ticket subsample the LLMs ran on.

The core LLM comparison table would otherwise mix full-test classical numbers with
1000-ticket LLM numbers, which is uneven. We slice each baseline's existing test
predictions to the same 1000 tickets and report on a like-for-like subsample.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sklearn.metrics import f1_score

from ticket_routing.data.loaders import build_loader_from_config
from ticket_routing.data.splitters import (
    stratified_test_subsample_for_llm,
    stratified_three_way_split,
)
from ticket_routing.evaluation.calibration import expected_calibration_error
from ticket_routing.utils.config import load_config


def slice_to_subsample(preds, confs, labels, full_test_labels, sub_texts, sub_labels):
    """Find indices of sub_labels in full_test_labels (preserving order)."""
    # The subsample preserves stratified ordering; need to find which indices were picked.
    # Since the subsample is deterministic (seed 42), we replay it to get indices.
    # Easier: use the texts as keys.
    return None  # see main: we reuse full_test_texts


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
    sub_texts, sub_labels = stratified_test_subsample_for_llm(
        split.test_texts, split.test_labels,
        max_total=1000, seed=cfg.dataset.random_seed,
    )
    # Map sub_texts back to indices in full test split
    text_to_idx = {t: i for i, t in enumerate(split.test_texts)}
    sub_indices = [text_to_idx[t] for t in sub_texts]
    print(f"Resolved {len(sub_indices)} of {len(sub_texts)} subsample tickets to "
          f"full-test indices")

    # Load each baseline's saved per-ticket data
    bundles = {
        "tfidf_logreg": "outputs/paper_headline_classical__20260521_153331/paper_bundle/raw/tfidf_logreg__classical__model_reported.json",
        "tfidf_svm": "outputs/paper_classical_extras__20260521_225728/paper_bundle/raw/tfidf_svm__classical__model_reported.json",
        "tfidf_random_forest": "outputs/paper_classical_extras__20260521_225728/paper_bundle/raw/tfidf_random_forest__classical__model_reported.json",
        "distilbert_finetuned": "outputs/paper_distilbert__20260521_230540/paper_bundle/raw/distilbert_finetuned__classical__model_reported.json",
    }

    print(f"\n{'Baseline':<24} {'Acc':>6} {'Macro-F1':>9} {'ECE-raw':>8}")
    print("-" * 50)
    for name, path in bundles.items():
        data = json.loads(Path(path).read_text())
        preds_full = data["predictions"]
        confs_full = data["confidences"]
        # Slice to subsample
        preds_sub = [preds_full[i] for i in sub_indices]
        confs_sub = [confs_full[i] for i in sub_indices]
        # Compute metrics
        correct = sum(1 for p, t in zip(preds_sub, sub_labels) if p == t)
        acc = correct / len(sub_labels)
        macro = f1_score(sub_labels, preds_sub, average="macro", zero_division=0)
        correctness = [1 if p == t else 0 for p, t in zip(preds_sub, sub_labels)]
        ece = expected_calibration_error(confs_sub, correctness, n_bins=10)["ece"]
        print(f"{name:<24} {acc:>6.3f} {macro:>9.3f} {ece:>8.3f}")


if __name__ == "__main__":
    main()
