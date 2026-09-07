#!/usr/bin/env python3
"""Create minimal LLM analysis inputs without legacy policy-selection metadata.

The review-time result JSON contains valid frozen predictions and confidence
vectors, but it also contains superseded test-minimizing summary metadata. This
script retains only fields needed to recompute camera-ready confidence and fixed
policy results. It performs no numerical transformation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def minimal_record(source: dict, source_name: str) -> dict:
    always_route = [
        row
        for row in source["cost"]
        if row["policy"] == "always_route" and row["wrong_auto_route_cost"] == 5.0
    ]
    if len(always_route) != 1:
        raise ValueError(f"{source_name}: expected one always-route row at wrong cost 5")
    classification = source["classification"]
    return {
        "schema_version": "camera-ready-llm-input-1.0",
        "provenance": {
            "review_time_filename": source_name,
            "operation": "field projection only; predictions, scores, and costs unchanged",
            "excluded": "legacy test-minimizing summaries and unneeded report fields",
        },
        "predictor_name": source["predictor_name"],
        "setting": source["setting"],
        "confidence_method": source["confidence_method"],
        "predictions": source["predictions"],
        "confidences": source["confidences"],
        "parse_status": source["parse_status"],
        "classification": {
            "n": classification["n"],
            "accuracy": classification["accuracy"],
            "macro_f1": classification["macro_f1"],
            "weighted_f1": classification["weighted_f1"],
            "parse_error_rate": classification["parse_error_rate"],
            "invalid_label_rate": classification["invalid_label_rate"],
        },
        "calibration": {
            "ece": source["calibration"]["ece"],
            "n_bins": source["calibration"]["n_bins"],
        },
        "cost": [always_route[0]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace only the projected JSON files in an existing output directory.",
    )
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.replace_existing:
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(source_dir.glob("*.json"))
    if len(sources) != 27:
        raise ValueError(f"Expected 27 source JSON files, found {len(sources)}")
    for source_path in sources:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        projected = minimal_record(source, source_path.name)
        (output_dir / source_path.name).write_text(
            json.dumps(projected, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {len(sources)} sanitized inputs to {output_dir}")


if __name__ == "__main__":
    main()
