"""Alias for run_classification.py.

The headline pipeline already performs abstention evaluation at every threshold
across every confidence method. This script forwards to that runner so the
filename hierarchy matches the spec, without duplicating logic.
"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().with_name("run_classification.py")), run_name="__main__")
