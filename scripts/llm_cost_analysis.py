"""Evaluate the fixed cost-derived LLM policy on frozen test predictions.

The break-even threshold is derived analytically from the configured costs; this
script never chooses a threshold from test outcomes. Raw verbalized confidence is
not treated as a probability for deployment-policy selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_INPUTS = REPO_ROOT / "artifacts" / "ictai2026" / "llm_inputs"


def cost(confs, correctness, tau, c_ok=0.0, c_triage=1.0, c_wrong=5.0):
    scores = np.asarray(confs, dtype=float)
    correct = np.asarray(correctness, dtype=bool)
    if scores.shape != correct.shape:
        raise ValueError("confidence and correctness vectors must have equal length")
    routed = scores >= tau
    per = np.where(routed, np.where(correct, c_ok, c_wrong), c_triage)
    return float(per.mean()), per


def coverage(confs, tau):
    return sum(1 for c in confs if c >= tau) / len(confs)


def correctness_from_raw_result(record):
    rows = [
        row
        for row in record["cost"]
        if row["policy"] == "always_route" and row["wrong_auto_route_cost"] == 5.0
    ]
    if len(rows) != 1:
        raise RuntimeError("cannot identify the canonical always-route cost vector")
    costs = np.asarray(rows[0]["per_ticket_costs"], dtype=float)
    if not np.all(np.isin(costs, [0.0, 5.0])):
        raise RuntimeError("unexpected values in always-route per-ticket costs")
    return costs == 0.0


def analyze(name, raw_path, iso_path, tau=0.8):
    raw = json.loads(Path(raw_path).read_text())
    iso = json.loads(Path(iso_path).read_text())
    iso_confs = iso["confidences"]
    if raw["predictions"] != iso["predictions"]:
        raise RuntimeError("predictions diverge between raw and isotonic artifacts")
    correctness = correctness_from_raw_result(raw)

    always, _ = cost(iso_confs, correctness, tau=0.0)
    fixed_cost, _ = cost(iso_confs, correctness, tau)
    fixed_cov = coverage(iso_confs, tau)
    n_routed = sum(1 for c in iso_confs if c >= tau)

    # Also include policy = always-route, agreement_geq_9 is in the JSON
    print(f"\n=== {name} ===")
    print(f"  always_route: E[cost] = {always:.4f} (cov 100.0%)")
    print(
        f"  ISO fixed tau={tau:.2f}: E[cost]={fixed_cost:.4f}, "
        f"routed={n_routed}/{len(iso_confs)} ({fixed_cov*100:.1f}%)"
    )
    rel_always = (always - fixed_cost) / always * 100
    rel_defer = (1.0 - fixed_cost) * 100
    print(f"  Δ: vs always-route={rel_always:+.1f}%, vs always-defer={rel_defer:+.1f}%")
    return {
        "name": name, "always": always,
        "threshold": tau,
        "n_test": len(iso_confs),
        "iso_cost": fixed_cost,
        "iso_cov": fixed_cov,
        "n_routed": n_routed,
        "relative_vs_always_route": rel_always,
        "relative_vs_always_defer": rel_defer,
    }


def main():
    raw_dir = LLM_INPUTS
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
            raise FileNotFoundError(f"required frozen inputs missing for {base}")
        r = analyze(display, raw_p, iso_p)
        all_results.append(r)

    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'LLM':<32} {'always':>7} {'fixed iso policy':>28}")
    for r in all_results:
        fixed = (
            f"τ={r['threshold']:.2f} {r['iso_cost']:.3f}; "
            f"{r['n_routed']}/{r['n_test']} routed"
        )
        print(f"{r['name']:<32} {r['always']:>7.3f} {fixed:>28}")


if __name__ == "__main__":
    main()
