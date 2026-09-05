"""Single entry point that produces a paper-ready bundle from results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from . import paper_notes, plots, tables
from ..utils.config import dump_config


def build_paper_bundle(
    out_dir: Path,
    cfg,
    dataset_bundle,
    split_bundle,
    results,
    env: dict,
) -> List[str]:
    bundle_dir = out_dir / "paper_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Tables ----------------------------------------------------------------
    tbls = {
        "table_1_dataset_summary": tables.dataset_summary_table(dataset_bundle, split_bundle),
        "table_2_classification": tables.classification_table(results),
        "table_3_calibration": tables.calibration_table(results),
        "table_4_abstention": tables.abstention_table(results),
        "table_5_cost": tables.cost_table(results),
        "table_6_error_analysis": tables.error_analysis_table(results),
    }
    tables.save_tables(bundle_dir / "tables", tbls)
    (bundle_dir / "paper_tables.md").write_text(tables.tables_as_markdown(tbls))

    # Figures ---------------------------------------------------------------
    fig_dir = bundle_dir / "figures"
    plots.save_pipeline_diagram(fig_dir)
    plots.save_class_distribution(fig_dir, dataset_bundle.labels)
    plots.save_coverage_accuracy_curve(fig_dir, results)
    plots.save_cost_threshold_curve(fig_dir, results, cfg.cost.default_wrong_auto_route)

    # Confusion matrix: best model without abstention.
    if results:
        best = max(results, key=lambda r: r.classification["accuracy"])
        plots.save_confusion_matrix(
            fig_dir,
            cm=best.classification["confusion_matrix"],
            labels=best.classification["per_class"]["labels"],
            filename="figure_5_confusion_best.png",
            title=f"Confusion: {best.predictor_name}/{best.setting} (no abstention)",
        )

    (bundle_dir / "paper_figures.md").write_text(
        "# Figures\n\n"
        "1. Pipeline diagram (`figures/figure_1_pipeline.mmd`)\n"
        "2. Class distribution (`figures/figure_2_class_distribution.png`)\n"
        "3. Coverage vs accuracy (`figures/figure_3_coverage_accuracy.png`)\n"
        "4. Cost vs threshold (`figures/figure_4_cost_threshold.png`)\n"
        "5. Confusion matrix (best model, no abstention) (`figures/figure_5_confusion_best.png`)\n"
        "6. Confusion matrix on auto-routed tickets (`figures/figure_6_confusion_best_abstention.png` — "
        "produced when the best abstention policy has enough auto-routed examples)\n"
    )

    # Section notes ---------------------------------------------------------
    paper_notes.write_result_summary(bundle_dir, results, env)
    paper_notes.write_abstract_notes(bundle_dir, results)
    paper_notes.write_methods_notes(bundle_dir, cfg, dict(dataset_bundle.metadata))
    paper_notes.write_results_notes(bundle_dir, results)
    paper_notes.write_limitations_notes(bundle_dir)
    paper_notes.write_reproducibility_statement(
        bundle_dir,
        cfg,
        {**dataset_bundle.metadata, "dataset_hash": dataset_bundle.dataset_hash},
        env,
    )
    paper_notes.write_threat_to_validity(bundle_dir)

    # Persist config snapshot alongside outputs.
    dump_config(cfg, bundle_dir / "experiment_config.yaml")

    # Raw JSON dumps for downstream re-analysis / bootstrap re-runs.
    raw_dir = bundle_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        slug = f"{r.predictor_name}__{r.setting}__{r.confidence_method}".replace(" ", "_")
        (raw_dir / f"{slug}.json").write_text(
            json.dumps(
                {
                    "predictor_name": r.predictor_name,
                    "setting": r.setting,
                    "confidence_method": r.confidence_method,
                    "classification": _jsonable(r.classification),
                    "calibration": _jsonable(r.calibration),
                    "abstention": _jsonable(r.abstention),
                    "cost": _jsonable(r.cost),
                    "error_analysis": r.error_analysis,
                    "confidences": r.confidences,
                    "predictions": r.predictions,
                    "parse_status": r.parse_status,
                    "metadata": _jsonable(r.metadata),
                },
                indent=2,
            )
        )

    # Generated-artifact manifest for the output checklist.
    artifacts: List[str] = []
    for p in sorted(bundle_dir.rglob("*")):
        if p.is_file():
            artifacts.append(str(p.relative_to(out_dir)))
    paper_notes.write_output_checklist(bundle_dir, artifacts)

    return artifacts


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj
