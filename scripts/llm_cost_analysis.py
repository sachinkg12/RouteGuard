"""Quantify cost-aware abstention for each LLM (raw and isotonic-recalibrated).

Addresses R2 #1: the claim that LLM cost-minimizing policies collapse to
defer-most needs a table, not just prose. Computes per-LLM minimum
expected cost across thresholds on the 1000-ticket subsample, under both
raw and isotonic confidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np


def cost(confs, preds, trues, tau, c_ok=0.0, c_triage=1.0, c_wrong=5.0):
    per = []
    for c, p, t in zip(confs, preds, trues):
        per.append(c_triage if c < tau else (c_ok if p == t else c_wrong))
    return sum(per) / len(per), np.array(per)


def coverage(confs, tau):
    return sum(1 for c in confs if c >= tau) / len(confs)


def analyze(name, raw_path, iso_path, trues):
    raw = json.loads(Path(raw_path).read_text())
    iso = json.loads(Path(iso_path).read_text())
    raw_confs = raw["confidences"]
    iso_confs = iso["confidences"]
    preds = raw["predictions"]
    assert preds == iso["predictions"], "predictions diverge"

    always, _ = cost(raw_confs, preds, trues, tau=0.0)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = {"raw": {}, "iso": {}}
    for label, confs in [("raw", raw_confs), ("iso", iso_confs)]:
        for tau in thresholds:
            c, _ = cost(confs, preds, trues, tau)
            cov = coverage(confs, tau)
            results[label][tau] = (c, cov)
    # Best across thresholds (excluding always-route)
    best_raw_tau = min(results["raw"], key=lambda t: results["raw"][t][0])
    best_iso_tau = min(results["iso"], key=lambda t: results["iso"][t][0])
    best_raw_cost, best_raw_cov = results["raw"][best_raw_tau]
    best_iso_cost, best_iso_cov = results["iso"][best_iso_tau]

    # Also include policy = always-route, agreement_geq_9 is in the JSON
    print(f"\n=== {name} ===")
    print(f"  always_route: E[cost] = {always:.4f} (cov 100.0%)")
    print(f"  RAW: best tau={best_raw_tau} -> E[cost]={best_raw_cost:.4f}, cov={best_raw_cov*100:.1f}%")
    print(f"  ISO: best tau={best_iso_tau} -> E[cost]={best_iso_cost:.4f}, cov={best_iso_cov*100:.1f}%")
    rel_raw = (always - best_raw_cost) / always * 100
    rel_iso = (always - best_iso_cost) / always * 100
    print(f"  Δ vs always: raw={rel_raw:+.1f}%, iso={rel_iso:+.1f}%")
    return {
        "name": name, "always": always,
        "raw_tau": best_raw_tau, "raw_cost": best_raw_cost, "raw_cov": best_raw_cov, "raw_rel": rel_raw,
        "iso_tau": best_iso_tau, "iso_cost": best_iso_cost, "iso_cov": best_iso_cov, "iso_rel": rel_iso,
    }


def main():
    from ticket_routing.data.loaders import build_loader_from_config
    from ticket_routing.data.splitters import (
        stratified_test_subsample_for_llm, stratified_three_way_split,
    )
    from ticket_routing.utils.config import load_config
    cfg = load_config("configs/experiment_default.yaml")
    bundle = build_loader_from_config(cfg.dataset).load()
    split = stratified_three_way_split(
        bundle,
        train_fraction=cfg.split.train_fraction,
        calibration_fraction=cfg.split.calibration_fraction,
        test_fraction=cfg.split.test_fraction,
        seed=cfg.dataset.random_seed,
    )
    sub_texts, sub_labels = stratified_test_subsample_for_llm(
        split.test_texts, split.test_labels, max_total=1000, seed=42,
    )
    print(f"Subsample size: {len(sub_labels)}")

    raw_dir = Path("outputs/paper_llm_4way_isotonic_retry__20260522_000630/paper_bundle/raw")
    variants = [
        ("Anthropic Haiku 4.5 (few-shot)", "haiku_few_shot_k3"),
        ("GPT-4o-mini (few-shot)",     "gpt4o_mini_few_shot_k3"),
        ("Qwen2.5-7B (few-shot)",      "qwen_few_shot_k3"),
        ("Llama-3-8B-Lite (zero-shot)","llama_zero_shot"),
        ("GPT-4o-mini (zero-shot)",    "gpt4o_mini_zero_shot"),
        ("Anthropic Haiku 4.5 (zero-shot)","haiku_zero_shot"),
        ("Qwen2.5-7B (zero-shot)",     "qwen_zero_shot"),
        ("Llama-3-8B-Lite (few-shot)", "llama_few_shot_k3"),
    ]
    settings = {n: ("few_shot" if "few_shot" in n else "zero_shot") for _, n in variants}

    all_results = []
    for display, base in variants:
        setting = settings[base]
        raw_p = raw_dir / f"{base}__{setting}__model_reported.json"
        iso_p = raw_dir / f"{base}__{setting}__isotonic.json"
        if not raw_p.exists() or not iso_p.exists():
            print(f"MISSING {base}, skipping")
            continue
        r = analyze(display, raw_p, iso_p, sub_labels)
        all_results.append(r)

    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'LLM':<32} {'always':>7} {'raw_best':>15} {'iso_best':>15}")
    for r in all_results:
        raw_str = f"τ={r['raw_tau']} {r['raw_cost']:.3f} ({r['raw_rel']:+.1f}%)"
        iso_str = f"τ={r['iso_tau']} {r['iso_cost']:.3f} ({r['iso_rel']:+.1f}%)"
        print(f"{r['name']:<32} {r['always']:>7.3f} {raw_str:>15} {iso_str:>15}")


if __name__ == "__main__":
    main()
