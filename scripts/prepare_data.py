"""Verify a dataset can be loaded and report its summary stats.

Usage:
    python scripts/prepare_data.py --config configs/experiment_default.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Allow running without installing the package.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ticket_routing.data.loaders import build_loader_from_config
from ticket_routing.data.splitters import stratified_three_way_split
from ticket_routing.utils.config import load_config
from ticket_routing.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logger = get_logger("prepare_data")
    cfg = load_config(args.config)
    loader = build_loader_from_config(cfg.dataset)
    bundle = loader.load()
    split = stratified_three_way_split(
        bundle,
        train_fraction=cfg.split.train_fraction,
        calibration_fraction=cfg.split.calibration_fraction,
        test_fraction=cfg.split.test_fraction,
        seed=cfg.dataset.random_seed,
    )

    counts = Counter(bundle.labels)
    summary = {
        "dataset_name": bundle.metadata.get("dataset_name"),
        "n_examples": len(bundle),
        "n_classes": len(bundle.label_names),
        "label_names": bundle.label_names,
        "label_counts": dict(counts),
        "train_size": len(split.train_texts),
        "calibration_size": len(split.calib_texts),
        "test_size": len(split.test_texts),
        "dataset_hash": bundle.dataset_hash,
    }
    logger.info("Dataset summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
