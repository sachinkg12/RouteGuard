"""Retrain DistilBERT and save calibration + test predictions for per-model tau* selection."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ticket_routing.data.loaders import build_loader_from_config
from ticket_routing.data.splitters import stratified_three_way_split
from ticket_routing.models.distilbert_classifier import DistilBertPredictor
from ticket_routing.utils.config import load_config


def main():
    cfg = load_config("configs/experiment_distilbert.yaml")
    bundle = build_loader_from_config(cfg.dataset).load()
    split = stratified_three_way_split(
        bundle,
        train_fraction=cfg.split.train_fraction,
        calibration_fraction=cfg.split.calibration_fraction,
        test_fraction=cfg.split.test_fraction,
        seed=cfg.dataset.random_seed,
    )

    t0 = time.time()
    print(f"Train: {len(split.train_texts)}, calib: {len(split.calib_texts)}, "
          f"test: {len(split.test_texts)}")

    clf = DistilBertPredictor(
        model_name="distilbert-base-uncased",
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        weight_decay=0.01,
        max_seq_length=256,
        warmup_ratio=0.06,
        random_state=cfg.dataset.random_seed,
        fp16=False,
        output_dir=".distilbert_train_calib",
    )
    clf.fit(split.train_texts, split.train_labels, split.label_names)
    print(f"Trained in {(time.time() - t0):.1f}s")

    calib_batch = clf.predict(split.calib_texts)
    test_batch = clf.predict(split.test_texts)
    print(f"Predicted calib+test in {(time.time() - t0):.1f}s total")

    out = Path("outputs/distilbert_calib_test_preds.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "calib_predictions": calib_batch.predicted_labels,
        "calib_confidences": [float(c) for c in calib_batch.confidence_scores],
        "calib_labels": list(split.calib_labels),
        "test_predictions": test_batch.predicted_labels,
        "test_confidences": [float(c) for c in test_batch.confidence_scores],
        "test_labels": list(split.test_labels),
    }))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
