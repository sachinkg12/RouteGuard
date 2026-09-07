#!/usr/bin/env python3
"""Build the sanitized common-subset inputs used to reproduce paper Table II.

Review-time raw reports include valid prediction vectors but also diagnostic
example ticket text. This projection keeps only labels, predictions,
confidences, and source hashes. It performs no numerical transformation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ticket_routing.data.loaders import build_loader_from_config
from ticket_routing.data.splitters import (
    stratified_test_subsample_for_llm,
    stratified_three_way_split,
)
from ticket_routing.utils.config import load_config


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "tfidf_logreg": REPO_ROOT / "outputs/paper_headline_classical__20260521_153331/paper_bundle/raw/tfidf_logreg__classical__model_reported.json",
    "tfidf_svm": REPO_ROOT / "outputs/paper_classical_extras__20260521_225728/paper_bundle/raw/tfidf_svm__classical__model_reported.json",
    "tfidf_random_forest": REPO_ROOT / "outputs/paper_classical_extras__20260521_225728/paper_bundle/raw/tfidf_random_forest__classical__model_reported.json",
    "distilbert_finetuned": REPO_ROOT / "outputs/paper_distilbert__20260521_230540/paper_bundle/raw/distilbert_finetuned__classical__model_reported.json",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/ictai2026/table_ii_common_subset.json",
    )
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.replace_existing:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")

    cfg = load_config(REPO_ROOT / "configs/experiment_default.yaml")
    bundle = build_loader_from_config(cfg.dataset).load()
    split = stratified_three_way_split(
        bundle,
        train_fraction=cfg.split.train_fraction,
        calibration_fraction=cfg.split.calibration_fraction,
        test_fraction=cfg.split.test_fraction,
        seed=cfg.dataset.random_seed,
    )
    sub_texts, sub_labels = stratified_test_subsample_for_llm(
        split.test_texts,
        split.test_labels,
        max_total=1000,
        seed=cfg.dataset.random_seed,
    )
    text_to_idx = {text: index for index, text in enumerate(split.test_texts)}
    if len(text_to_idx) != len(split.test_texts):
        raise ValueError("full test text is not unique; index mapping would be ambiguous")
    sub_indices = [text_to_idx[text] for text in sub_texts]

    models = {}
    for name, source_path in SOURCES.items():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        predictions = source["predictions"]
        confidences = source["confidences"]
        if len(predictions) != len(split.test_labels) or len(confidences) != len(split.test_labels):
            raise ValueError(f"{name}: saved vectors do not match configured test length")
        models[name] = {
            "review_time_source": str(source_path.relative_to(REPO_ROOT)),
            "review_time_source_sha256": file_hash(source_path),
            "predictions": [predictions[index] for index in sub_indices],
            "confidences": [confidences[index] for index in sub_indices],
        }

    payload = {
        "schema_version": "camera-ready-table-ii-input-1.0",
        "provenance": {
            "operation": "deterministic subsample and field projection only",
            "excluded": "ticket text and unneeded review-time report fields",
            "split_seed": cfg.dataset.random_seed,
            "full_test_n": len(split.test_labels),
            "subsample_n": len(sub_labels),
        },
        "true_labels": list(sub_labels),
        "models": models,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote sanitized Table II inputs for {len(models)} models to {output}")


if __name__ == "__main__":
    main()
