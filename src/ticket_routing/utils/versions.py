"""Capture python + package + git versions for the reproducibility statement."""
from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from typing import Optional


_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "pyyaml",
    "pydantic",
    "openai",
    "transformers",
    "torch",
)


def _safe_version(pkg: str) -> Optional[str]:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def git_commit() -> Optional[str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit or None
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: _safe_version(p) for p in _TRACKED_PACKAGES},
        "git_commit": git_commit(),
    }
