"""F7 (ECE bootstrap CIs) + F8 (paired accuracy-gap CI/McNemar) from bundle raw JSON."""
import json, os, sys
import numpy as np
sys.path.insert(0, "src")
from ticket_routing.evaluation.calibration import expected_calibration_error

B = "outputs/paper_llm_4way_isotonic_retry__20260522_000630/paper_bundle/raw"
NBOOT = 1000
SEED = 1337

def load(name):
    return json.load(open(os.path.join(B, name)))

def correctness_from_costs(d):
    # always_route policy (index 0): per-ticket cost 0.0 == correct, >0 == wrong
    ar = d["cost"][0]
    assert ar["policy"] == "always_route", ar["policy"]
    costs = np.asarray(ar["per_ticket_costs"], dtype=float)
    return (costs == 0.0).astype(int)

def ece_point(conf, corr):
    return expected_calibration_error(conf, corr)["ece"]

def ece_ci(conf, corr, rng):
    conf = np.asarray(conf, float); corr = np.asarray(corr, int)
    n = len(conf); vals = np.empty(NBOOT)
    for b in range(NBOOT):
        idx = rng.integers(0, n, n)
        vals[b] = ece_point(conf[idx], corr[idx])
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

PRED = ["gpt4o_mini_zero_shot","gpt4o_mini_few_shot_k3","haiku_zero_shot","haiku_few_shot_k3",
        "qwen_zero_shot","qwen_few_shot_k3","llama_zero_shot","llama_few_shot_k3","tfidf_logreg"]
SETTING = {p: ("classical" if p=="tfidf_logreg" else ("few_shot" if "few_shot" in p else "zero_shot")) for p in PRED}
METHODS = ["model_reported","agreement","isotonic"]

rng = np.random.default_rng(SEED)
print("=== F7: ECE point (validate) + 95% bootstrap CI ===")
print(f"{'predictor':24s} {'method':14s} {'ECE':>7s} {'CI_low':>7s} {'CI_high':>7s}")
for p in PRED:
    for m in METHODS:
        fn = f"{p}__{SETTING[p]}__{m}.json"
        d = load(fn)
        conf = d["confidences"]; corr = correctness_from_costs(d)
        pt = ece_point(conf, corr)
        lo, hi = ece_ci(conf, corr, rng)
        print(f"{p:24s} {m:14s} {pt:7.3f} {lo:7.3f} {hi:7.3f}")

print("\n=== F8: paired accuracy gap, TF-IDF+LR vs best LLM (Haiku few-shot), common 1000 ===")
c = correctness_from_costs(load("tfidf_logreg__classical__model_reported.json"))
l = correctness_from_costs(load("haiku_few_shot_k3__few_shot__model_reported.json"))
acc_c, acc_l = c.mean(), l.mean()
gap = acc_c - acc_l
rng2 = np.random.default_rng(SEED)
n = len(c); diffs = np.empty(NBOOT)
for b in range(NBOOT):
    idx = rng2.integers(0, n, n)
    diffs[b] = c[idx].mean() - l[idx].mean()
lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
# McNemar on paired correctness
b01 = int(np.sum((c==0) & (l==1)))  # classical wrong, llm right
b10 = int(np.sum((c==1) & (l==0)))  # classical right, llm wrong
from scipy.stats import binomtest
mc = binomtest(min(b01,b10), b01+b10, 0.5)
print(f"classical acc={acc_c:.3f}  llm acc={acc_l:.3f}  gap={gap:.3f}")
print(f"paired bootstrap 95% CI on gap = [{lo:.3f}, {hi:.3f}]")
print(f"McNemar discordant: classical-only-right={b10}, llm-only-right={b01}, p={mc.pvalue:.2e}")
