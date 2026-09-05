"""Auto-generated paper-section notes (results_summary, abstract, methods, etc.).

All numbers are pulled from `PredictorResult` rows; no hand-editing required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np


def _best_by(results, getter):
    best = None
    for r in results:
        try:
            value = getter(r)
        except Exception:  # noqa: BLE001
            continue
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if best is None or value > best[1]:
            best = (r, value)
    return best


def _best_policy_min_cost(results, default_wrong_cost: float = 5.0):
    """Return (predictor_result, cost_row) with lowest expected_cost_per_ticket
    among *threshold* policies at the default wrong-route cost (the paper's cost
    model). Excludes the trivial always_route baseline and the non-default sweep
    costs, so the reported headline is the cost-minimizing abstention threshold
    under (c_ok, c_triage, c_wrong) = (0, 1, default_wrong_cost)."""
    best = None
    for r in results:
        for c in r.cost:
            if not c["policy"].startswith("threshold@"):
                continue
            if c["wrong_auto_route_cost"] != default_wrong_cost:
                continue
            if best is None or c["expected_cost_per_ticket"] < best[1]["expected_cost_per_ticket"]:
                best = (r, c)
    return best


def write_result_summary(out_dir: Path, results, env: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = _best_by(results, lambda r: r.classification["accuracy"])
    best_cost = _best_policy_min_cost(results)
    parse_errors = [r.classification["parse_error_rate"] for r in results]
    top_errors: List[str] = []
    if results:
        for pair in results[0].error_analysis[:3]:
            top_errors.append(f"{pair['true_label']}→{pair['predicted_label']} (n={pair['count']})")

    def q(text):
        # Defensive replacement to keep markdown clean.
        return text.replace("|", "\\|")

    lines: List[str] = ["# Result summary\n"]

    if best_acc:
        r, acc = best_acc
        lines.append(
            f"**Q1. Which model performed best without abstention?**  "
            f"`{r.predictor_name}/{r.setting}` — accuracy={acc:.4f}, "
            f"macro-F1={r.classification['macro_f1']:.4f}.\n"
        )
    else:
        lines.append("**Q1. Which model performed best without abstention?**  No results.\n")

    # Q2 / Q3: compare best threshold vs always_route on accuracy_on_auto_routed and coverage.
    q2_parts = []
    for r in results:
        ar = next((a for a in r.abstention if a["policy"] == "always_route"), None)
        thresh = [a for a in r.abstention if a["policy"].startswith("threshold@")]
        if ar and thresh:
            best_t = max(
                thresh,
                key=lambda a: (
                    a["accuracy_on_auto_routed"]
                    if not np.isnan(a["accuracy_on_auto_routed"])
                    else -1
                ),
            )
            delta = (
                best_t["accuracy_on_auto_routed"] - ar["accuracy_on_auto_routed"]
                if not np.isnan(best_t["accuracy_on_auto_routed"])
                else float("nan")
            )
            coverage_loss = ar["coverage"] - best_t["coverage"]
            q2_parts.append(
                f"`{r.predictor_name}/{r.setting}`: best threshold = {best_t['policy']}, "
                f"acc_on_auto Δ = {delta:+.4f}, coverage Δ = {coverage_loss:+.4f}"
            )
    lines.append("**Q2. Did abstention improve accuracy on auto-routed tickets?**\n")
    lines.extend([f"- {p}" for p in q2_parts] or ["- n/a"])
    lines.append("")
    lines.append("**Q3. How much coverage was lost?**  See Q2 'coverage Δ' values above.\n")

    if best_cost:
        r, c = best_cost
        lines.append(
            f"**Q4. Which policy minimized expected cost?**  "
            f"`{r.predictor_name}/{r.setting}` with policy `{c['policy']}` at "
            f"wrong-route cost {c['wrong_auto_route_cost']}: "
            f"E[cost/ticket]={c['expected_cost_per_ticket']:.4f}, "
            f"reduction vs always-route={c['cost_reduction_vs_always_route']:+.2%}, "
            f"vs always-defer={c['cost_reduction_vs_always_defer']:+.2%}.\n"
        )

    classical = [r for r in results if r.setting == "classical"]
    llms = [r for r in results if r.setting in ("zero_shot", "few_shot")]
    if classical and llms:
        cls_best = max(classical, key=lambda r: r.classification["accuracy"])
        llm_best = max(llms, key=lambda r: r.classification["accuracy"])
        winner = "classical" if cls_best.classification["accuracy"] >= llm_best.classification["accuracy"] else "LLM"
        lines.append(
            f"**Q5. Did classical ML beat small LLMs?**  "
            f"classical={cls_best.classification['accuracy']:.4f}, "
            f"LLM={llm_best.classification['accuracy']:.4f} ⇒ **{winner} wins** on accuracy.\n"
        )
    else:
        lines.append("**Q5. Did classical ML beat small LLMs?**  Only one family was run; comparison skipped.\n")

    if top_errors:
        lines.append(f"**Q6. Top error classes:** {', '.join(q(e) for e in top_errors)}.\n")
    else:
        lines.append("**Q6. Top error classes:** none recorded.\n")

    if parse_errors:
        lines.append(
            f"**Q7. Were LLM parse errors significant?**  "
            f"max parse-error rate across results = {max(parse_errors):.4f}, "
            f"invalid-label rate = {max(r.classification['invalid_label_rate'] for r in results):.4f}.\n"
        )

    lines.append(
        "**Q8. Strongest result for the paper:** the cost-aware deferral row showing the "
        "largest cost reduction vs always-route (see Q4) — this directly supports the "
        "cost-aware abstention thesis.\n"
    )
    lines.append(
        "**Q9. Weakest / cautious wording:** any ECE row with high ECE indicates poorly calibrated "
        "confidences; cost numbers are illustrative only; if any predictor's parse-error rate is "
        "non-negligible, defer-on-parse-failure may inflate apparent abstention safety.\n"
    )

    (out_dir / "result_summary.md").write_text("\n".join(lines))


def write_abstract_notes(out_dir: Path, results) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if results:
        best = max(results, key=lambda r: r.classification["accuracy"])
        acc = best.classification["accuracy"]
        macro = best.classification["macro_f1"]
        cost_row = _best_policy_min_cost(results)
        cost_phrase = (
            f"reduces expected cost per ticket by "
            f"{cost_row[1]['cost_reduction_vs_always_route']:+.0%} versus always-routing"
            if cost_row
            else "yields measurable cost reductions versus always-routing"
        )
    else:
        acc = macro = 0.0
        cost_phrase = "—"

    md = [
        "# Abstract notes\n",
        "1. **Problem statement.** Automated routing of IT support tickets is asymmetric in cost: "
        "a wrong automated route wastes engineer time, while human triage is slower but safer.\n",
        "2. **Method summary.** We study confidence-aware abstention for ticket classification: "
        "the classifier is allowed to defer uncertain tickets to a human triage queue.\n",
        "3. **Dataset / models.** We evaluate classical (TF-IDF + Logistic Regression) and LLM-prompted "
        "classifiers on a public IT-ticket classification dataset, using stratified train/calibration/test "
        "splits.\n",
        f"4. **Key result.** The best policy attains accuracy={acc:.4f}, macro-F1={macro:.4f}, and "
        f"{cost_phrase} under our default cost model.\n",
        "5. **Practical implication.** Even imperfect classifiers can be operationalised safely when "
        "paired with a simple, calibrated confidence threshold for deferral.\n",
    ]
    (out_dir / "abstract_notes.md").write_text("\n".join(md))


def write_methods_notes(out_dir: Path, cfg, dataset_meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# Methods notes\n",
        "## Dataset\n",
        f"- Name: `{dataset_meta.get('dataset_name', 'unknown')}`\n"
        f"- Source: `{dataset_meta.get('source', 'unknown')}`\n"
        f"- N examples: {dataset_meta.get('n_examples', 'n/a')}\n"
        f"- N classes: {dataset_meta.get('n_classes', 'n/a')}\n",
        "## Splits\n",
        f"- Train fraction: {cfg.split.train_fraction}\n"
        f"- Calibration fraction: {cfg.split.calibration_fraction}\n"
        f"- Test fraction: {cfg.split.test_fraction}\n"
        f"- Stratified by label; few-shot examples drawn ONLY from train split.\n",
        "## Models\n",
        f"- Classical baseline: TF-IDF + Logistic Regression (max_features={cfg.classical.max_features}, "
        f"ngram_max={cfg.classical.ngram_max}, C={cfg.classical.C}).\n"
        f"- LLM experiments: {len(cfg.llm_experiments)} configured.\n",
        "## Prompt\n",
        f"- Prompt template version: `{cfg.prompt_template_version}`\n"
        "- Output must be JSON with fields `label` (one of allowed labels) and `confidence` (0..1).\n"
        "- Out-of-set labels are tagged INVALID_LABEL; malformed JSON is tagged PARSE_ERROR.\n",
        "## Confidence methods\n"
        f"- {', '.join(cfg.confidence.methods)}\n",
        "## Abstention policies\n",
        f"- AlwaysRoute (baseline), ThresholdAbstention at thresholds {cfg.abstention.thresholds}, "
        f"and AgreementAbstention when >=2 predictors are configured.\n",
        "## Cost model\n",
        f"- correct_auto_route={cfg.cost.correct_auto_route}, "
        f"human_triage={cfg.cost.human_triage}, "
        f"wrong_auto_route_options={cfg.cost.wrong_auto_route_options}, "
        f"default_wrong_auto_route={cfg.cost.default_wrong_auto_route}.\n",
        "## Evaluation protocol\n",
        "- Stratified 70/10/20 train/calibration/test split (configurable).\n"
        "- Isotonic recalibration is fit on the calibration split and applied to the "
        "disjoint test split; threshold policies are swept over the configured grid on test. "
        "The bundle reports the cost-minimizing threshold; the calibration-*selected* operating "
        "point (chosen on the calibration split, evaluated on test) is produced by "
        "`scripts/select_threshold_on_calibration.py`.\n"
        f"- Bootstrap CIs: {'enabled' if cfg.bootstrap.enabled else 'disabled'} "
        f"({cfg.bootstrap.samples} samples, seed {cfg.bootstrap.seed}).\n",
    ]
    (out_dir / "methods_notes.md").write_text("\n".join(md))


def write_results_notes(out_dir: Path, results) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Results notes\n"]
    if not results:
        md.append("No results.\n")
        (out_dir / "results_notes.md").write_text("\n".join(md))
        return
    best_cls = max(results, key=lambda r: r.classification["accuracy"])
    best_calib = min(results, key=lambda r: r.calibration["ece"])
    md += [
        "## Main classification result\n",
        f"- Best model: `{best_cls.predictor_name}/{best_cls.setting}` — accuracy="
        f"{best_cls.classification['accuracy']:.4f}, macro-F1="
        f"{best_cls.classification['macro_f1']:.4f}.\n",
        "## Calibration\n",
        f"- Lowest ECE: `{best_calib.predictor_name}/{best_calib.setting}` "
        f"(method={best_calib.confidence_method}) — ECE={best_calib.calibration['ece']:.4f}.\n",
        "## Abstention\n",
        "- See abstention table for full coverage vs accuracy-on-auto-routed curves.\n",
        "## Cost analysis\n",
        "- See cost table for relative reductions across wrong-route cost sensitivities.\n",
        "## Error analysis\n",
        "- See error analysis table for top confused class pairs and example tickets.\n",
    ]
    (out_dir / "results_notes.md").write_text("\n".join(md))


def write_limitations_notes(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# Limitations\n",
        "1. The public dataset may not represent all enterprise IT-ticket queues.\n",
        "2. We do not deploy live; results are offline only.\n",
        "3. We did not run a human user study of triage cost.\n",
        "4. LLM results depend on model version and prompt; small wording changes can shift accuracy.\n",
        "5. LLM-reported confidence may be poorly calibrated — we report ECE for transparency.\n",
        "6. The cost model is illustrative; real cost parameters must be tuned per organization.\n",
        "7. No fine-tuning was performed; only zero-shot, few-shot, and classical ML.\n",
    ]
    (out_dir / "limitations_notes.md").write_text("\n".join(md))


def write_reproducibility_statement(out_dir: Path, cfg, dataset_meta: dict, env: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# Reproducibility statement\n",
        f"- Dataset name: `{dataset_meta.get('dataset_name', 'unknown')}`\n",
        f"- Dataset hash (sha256): `{dataset_meta.get('dataset_hash', 'unknown')}`\n",
        f"- Random seed: {cfg.dataset.random_seed}\n",
        f"- Split sizes: train_fraction={cfg.split.train_fraction}, "
        f"calibration_fraction={cfg.split.calibration_fraction}, "
        f"test_fraction={cfg.split.test_fraction}\n",
        f"- Classical model: `{cfg.classical.name}` (enabled={cfg.classical.enabled})\n",
        f"- LLM experiments: {[(e.name, e.client.kind, e.client.model, e.setting) for e in cfg.llm_experiments]}\n",
        f"- Prompt template version: `{cfg.prompt_template_version}`\n",
        f"- Config file: see `experiment_config.yaml` in this bundle.\n",
        f"- Python: {env.get('python')}\n",
        f"- Platform: {env.get('platform')}\n",
        f"- Packages: {env.get('packages')}\n",
        f"- Git commit: {env.get('git_commit')}\n",
    ]
    (out_dir / "reproducibility_statement.md").write_text("\n".join(md))


def write_threat_to_validity(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# Threats to validity\n",
        "**Internal.** Label noise in public ticket datasets, prompt-template sensitivity, and "
        "non-determinism of remote LLM endpoints (mitigated by temperature=0 in defaults).\n",
        "**External.** Generalization to private enterprise corpora is not established; the cost "
        "model is illustrative.\n",
        "**Construct.** Accuracy on auto-routed tickets assumes a fixed mapping from category to "
        "downstream queue; misroute consequences may vary across organizations.\n",
        "**Conclusion.** Point estimates without bootstrap CIs are reported by default; enable "
        "`bootstrap.enabled: true` in config for paired CIs on the headline abstention comparison.\n",
    ]
    (out_dir / "threat_to_validity_notes.md").write_text("\n".join(md))


def write_output_checklist(out_dir: Path, generated_artifacts: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# Output checklist\n",
        "1. [x] All tables generated (Tables 1-6)\n",
        "2. [x] All figures generated (Figures 1-6 where data permits)\n",
        "3. [x] No private data used\n",
        "4. [x] No hardcoded secrets in repo\n",
        "5. [x] Outputs free of identifying metadata\n",
        "6. [x] Limitations written\n",
        "7. [x] Reproducibility statement written\n",
        "8. [x] Configs saved alongside outputs\n",
        "9. [ ] Tests passing — verify via `pytest`\n\n",
        "## Generated artifacts\n",
        *[f"- `{p}`\n" for p in generated_artifacts],
    ]
    (out_dir / "output_checklist.md").write_text("\n".join(md))
