"""Rebuilds the paper bundle from a saved run directory.

Useful when you have raw JSON results and want to regenerate tables/figures
without rerunning predictors. Currently a thin wrapper that points users at
`run_classification.py` since the bundle is produced inline; this stub exists
to match the documented script surface and to host future re-bundling logic.

Usage:
    python scripts/make_paper_bundle.py --help
"""
from __future__ import annotations

import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=False,
        help="Saved run directory. If provided, prints its bundle path.",
    )
    args = parser.parse_args()
    if args.run_dir:
        print(f"Paper bundle: {args.run_dir.rstrip('/')}/paper_bundle/")
    else:
        print(
            "Paper bundles are produced by `python scripts/run_classification.py --config <yaml>`.\n"
            "They land under `outputs/<run_id>__<timestamp>/paper_bundle/`."
        )
        sys.exit(0)
