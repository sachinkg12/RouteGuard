"""Matplotlib figures for result reporting.

Figure 1 is a static pipeline diagram (Mermaid text). Figures 2-6 are PNGs.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# Shared color palette for report figures.
PAPER_TEAL = "#1F6F8B"      # bar fill, line accent
PAPER_NAVY = "#0F3D5C"      # primary line / annotation
PAPER_STROKE = "#1F3A5F"    # outlines
PAPER_GREY_GRID = "#D9D9D9" # subtle grid


def _apply_paper_style() -> None:
    """Apply a consistent styling to subsequent matplotlib figures."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "grid.color": PAPER_GREY_GRID,
        "grid.linewidth": 0.4,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


PIPELINE_MERMAID = """flowchart LR
    A[Ticket Text] --> B[Classifier]
    B --> C[Confidence Estimator]
    C --> D{Abstention Policy}
    D -->|AUTO_ROUTE| E[Automated Routing]
    D -->|DEFER_TO_HUMAN| F[Human Triage]
"""


def save_pipeline_diagram(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figure_1_pipeline.mmd").write_text(PIPELINE_MERMAID)
    _render_pipeline_png(out_dir / "figure_1_pipeline.png")


def _render_pipeline_png(path: Path) -> None:
    """Pipeline figure drawn directly with matplotlib (no Mermaid CLI)."""
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    nodes = [
        (0.5, "Ticket text"),
        (2.4, "Classifier"),
        (4.3, "Confidence\nestimator"),
        (6.2, "Abstention\npolicy"),
    ]
    for x, label in nodes:
        ax.add_patch(plt.Rectangle((x, 0.7), 1.5, 0.7, fill=True,
                                   facecolor="#e8eef7", edgecolor="#33506b", linewidth=1.4))
        ax.text(x + 0.75, 1.05, label, ha="center", va="center", fontsize=10, color="#1a1a1a")

    # Two outcome nodes from the abstention policy.
    ax.add_patch(plt.Rectangle((8.1, 1.15), 1.7, 0.55, fill=True,
                               facecolor="#dff3df", edgecolor="#2f7a2f", linewidth=1.4))
    ax.text(8.95, 1.43, "Auto route", ha="center", va="center", fontsize=10, color="#1a1a1a")

    ax.add_patch(plt.Rectangle((8.1, 0.30), 1.7, 0.55, fill=True,
                               facecolor="#fbe6cf", edgecolor="#a35a1d", linewidth=1.4))
    ax.text(8.95, 0.58, "Human triage", ha="center", va="center", fontsize=10, color="#1a1a1a")

    # Chain arrows.
    arrow_kw = dict(arrowstyle="->", color="#33506b", lw=1.4)
    for i in range(len(nodes) - 1):
        x0 = nodes[i][0] + 1.5
        x1 = nodes[i + 1][0]
        ax.annotate("", xy=(x1, 1.05), xytext=(x0, 1.05), arrowprops=arrow_kw)

    # Decision arrows to the two outcomes.
    x_dec = nodes[-1][0] + 1.5
    ax.annotate("", xy=(8.1, 1.42), xytext=(x_dec, 1.05), arrowprops=arrow_kw)
    ax.text(7.55, 1.55, "high conf.", ha="center", va="bottom", fontsize=8, color="#2f7a2f", style="italic")
    ax.annotate("", xy=(8.1, 0.58), xytext=(x_dec, 1.05), arrowprops=arrow_kw)
    ax.text(7.55, 0.45, "low conf.", ha="center", va="top", fontsize=8, color="#a35a1d", style="italic")

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_class_distribution(out_dir: Path, labels: Sequence[str]) -> None:
    """Horizontal bar chart, NotebookLM-inspired aesthetic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_paper_style()
    counts = Counter(labels)
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.42 * len(names))))
    bars = ax.barh(
        names[::-1], vals[::-1],
        color=PAPER_TEAL, edgecolor="none", height=0.7,
    )
    # Value labels just past each bar end.
    xmax = max(vals) if vals else 1
    for bar, v in zip(bars, vals[::-1]):
        ax.text(
            bar.get_width() + xmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{v:,}",
            ha="left", va="center", fontsize=8, color="#222222",
        )
    ax.set_xlabel("Number of tickets")
    ax.set_xlim(0, xmax * 1.15)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.4, color=PAPER_GREY_GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_2_class_distribution.png")
    plt.close(fig)


def save_coverage_accuracy_curve(out_dir: Path, results) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in results:
        xs, ys = [], []
        for a in r.abstention:
            if a["policy"].startswith("threshold@") and not np.isnan(
                a["accuracy_on_auto_routed"]
            ):
                xs.append(a["coverage"])
                ys.append(a["accuracy_on_auto_routed"])
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        if len(xs):
            ax.plot(xs, ys, marker="o", label=f"{r.predictor_name}/{r.setting}")
    ax.set_xlabel("Coverage (auto-route rate)")
    ax.set_ylabel("Accuracy on auto-routed tickets")
    ax.set_title("Coverage vs. accuracy")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_3_coverage_accuracy.png", dpi=150)
    plt.close(fig)


def save_cost_threshold_curve(out_dir: Path, results, default_wrong_cost: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in results:
        xs, ys = [], []
        for c in r.cost:
            if (
                c["policy"].startswith("threshold@")
                and c["wrong_auto_route_cost"] == default_wrong_cost
            ):
                xs.append(float(c["policy"].split("@", 1)[1]))
                ys.append(c["expected_cost_per_ticket"])
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        if len(xs):
            ax.plot(xs, ys, marker="o", label=f"{r.predictor_name}/{r.setting}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected cost per ticket")
    ax.set_title(f"Cost vs. threshold (wrong-route cost={default_wrong_cost})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_4_cost_threshold.png", dpi=150)
    plt.close(fig)


def save_confusion_matrix(
    out_dir: Path, cm: list[list[int]], labels: Sequence[str], filename: str, title: str
) -> None:
    """Sequential-Blues confusion matrix with thin white grid between cells."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_paper_style()
    cm_arr = np.asarray(cm, dtype=float)
    n = len(labels)
    side = max(4.0, 0.55 * n)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm_arr, cmap="Blues", aspect="equal")
    # Thin white grid between cells.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    # Major ticks for labels only.
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis="both", which="major", length=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # Strip outer spines (cells alone carry the visual story).
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Count", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2, width=0.4)
    cbar.outline.set_linewidth(0.4)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)
