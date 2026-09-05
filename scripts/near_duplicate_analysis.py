"""Near-duplicate audit + group-aware (leakage-free) re-evaluation of the
classical baselines.

Motivation: a single stratified *random* split can place near-duplicate or
template-like tickets in both train and test, which can inflate lexical
classifiers. This script quantifies exact/near-duplicate rates, measures how
much of the main random split leaks near-twins across train/test, and re-fits
the classical baselines under a *group-aware* split (near-duplicate clusters kept
on the same side) to show the classical-vs-LLM gap survives.

Run (uses the same loader/split as the main experiments):
    .venv/bin/python scripts/near_duplicate_analysis.py --config configs/experiment_default.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ticket_routing.data.loaders import build_loader_from_config  # noqa: E402
from ticket_routing.data.splitters import stratified_three_way_split  # noqa: E402
from ticket_routing.utils.config import load_config  # noqa: E402

NEAR_THRESHOLDS = [0.80, 0.90, 0.95]
PRIMARY = 0.90


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).lower().strip())


def classical_fit_eval(train_texts, train_labels, test_texts, test_labels,
                       max_features, clf_kind="lr"):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                          min_df=1, sublinear_tf=True)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    if clf_kind == "lr":
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=0)
    else:
        raise ValueError(clf_kind)
    clf.fit(Xtr, train_labels)
    pred = clf.predict(Xte)
    return (accuracy_score(test_labels, pred),
            f1_score(test_labels, pred, average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_default.yaml")
    ap.add_argument("--sim-features", type=int, default=30000,
                    help="TF-IDF features for near-dup similarity (speed knob)")
    ap.add_argument("--clf-features", type=int, default=100000,
                    help="TF-IDF features for the classifier (match main config=100000)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bundle = build_loader_from_config(cfg.dataset).load()
    texts = list(bundle.texts)
    labels = list(bundle.labels)
    n = len(texts)
    print(f"Corpus: {n} tickets, {len(set(labels))} classes")

    # ---- 1. exact duplicates (normalized text) ----
    norm = [_norm(t) for t in texts]
    counts = Counter(norm)
    exact_dup_rows = sum(c for c in counts.values() if c > 1) - sum(1 for c in counts.values() if c > 1)
    n_unique = len(counts)
    print("\n=== 1. EXACT DUPLICATES (normalized text) ===")
    print(f"  unique normalized texts : {n_unique}")
    print(f"  exact-duplicate tickets : {exact_dup_rows} ({100*exact_dup_rows/n:.2f}% of corpus)")

    # ---- 2. near-duplicate similarity graph ----
    t0 = time.time()
    print("\n=== 2. NEAR-DUPLICATES (TF-IDF cosine) ===")
    vec = TfidfVectorizer(max_features=args.sim_features, ngram_range=(1, 2),
                          min_df=1, sublinear_tf=True)
    X = vec.fit_transform(texts)  # L2-normalized -> cosine = dot
    nn = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X)
    dist, idx = nn.kneighbors(X)  # includes self at col 0
    sim = 1.0 - dist
    print(f"  kNN built in {time.time()-t0:.1f}s")

    for thr in NEAR_THRESHOLDS:
        rows, cols = [], []
        for i in range(n):
            for j_pos in range(1, idx.shape[1]):  # skip self
                j = idx[i, j_pos]
                if sim[i, j_pos] >= thr and j != i:
                    rows.append(i); cols.append(j)
        if rows:
            A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
            A = A + A.T
            ncomp, comp = connected_components(A, directed=False)
            csize = Counter(comp)
            in_cluster = sum(1 for c in comp if csize[c] > 1)
        else:
            comp = np.arange(n); in_cluster = 0
        tag = " <-- primary" if thr == PRIMARY else ""
        print(f"  sim>={thr:.2f}: {in_cluster} tickets have a near-twin "
              f"({100*in_cluster/n:.2f}% of corpus){tag}")
        if thr == PRIMARY:
            primary_comp = comp
            primary_in_cluster = in_cluster

    # ---- 3. leakage in the main random split ----
    print("\n=== 3. LEAKAGE IN THE MAIN RANDOM SPLIT (seed 42, 70/10/20) ===")
    split = stratified_three_way_split(
        bundle,
        train_fraction=cfg.split.train_fraction,
        calibration_fraction=cfg.split.calibration_fraction,
        test_fraction=cfg.split.test_fraction,
        seed=cfg.dataset.random_seed,
    )
    # map texts -> corpus index via first-occurrence lookup is unsafe with dups;
    # instead recompute tfidf per subset and query test->train nearest.
    Xtr = vec.transform(split.train_texts)
    Xte = vec.transform(split.test_texts)
    nn_tr = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn_tr.fit(Xtr)
    d_te, _ = nn_tr.kneighbors(Xte)
    sim_te = 1.0 - d_te.ravel()
    print(f"  test tickets: {len(split.test_texts)}")
    for thr in NEAR_THRESHOLDS:
        leak = int((sim_te >= thr).sum())
        tag = " <-- primary" if thr == PRIMARY else ""
        print(f"  test tickets with a train near-twin sim>={thr:.2f}: "
              f"{leak} ({100*leak/len(split.test_texts):.2f}% of test){tag}")

    # ---- 4. group-aware (leakage-free) re-evaluation ----
    print("\n=== 4. GROUP-AWARE SPLIT RE-EVALUATION (clusters kept together) ===")
    # random-split baseline (main pipeline) for reference
    acc_rand, f1_rand = classical_fit_eval(
        split.train_texts, split.train_labels, split.test_texts, split.test_labels,
        args.clf_features)
    print(f"  TF-IDF+LR on RANDOM split (main-style): acc={acc_rand:.4f} macroF1={f1_rand:.4f}")

    # group split by near-dup cluster (primary threshold); test_size=0.30 so the
    # train fraction (~70%) matches the main random split (only leakage differs)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    gtr, gte = next(gss.split(np.arange(n), groups=primary_comp))
    # verify zero shared clusters
    shared = set(primary_comp[gtr]) & set(primary_comp[gte])
    acc_grp, f1_grp = classical_fit_eval(
        [texts[i] for i in gtr], [labels[i] for i in gtr],
        [texts[i] for i in gte], [labels[i] for i in gte],
        args.clf_features)
    print(f"  TF-IDF+LR on GROUP split (leakage-free): acc={acc_grp:.4f} macroF1={f1_grp:.4f}")
    print(f"  shared clusters across group split: {len(shared)} (should be 0)")
    print(f"  group train/test sizes: {len(gtr)}/{len(gte)}")

    print("\n=== SUMMARY ===")
    print(f"  exact-dup rate      : {100*exact_dup_rows/n:.2f}%")
    print(f"  near-dup rate (>=.90): {100*primary_in_cluster/n:.2f}% of corpus")
    print(f"  random-split acc    : {acc_rand:.4f}")
    print(f"  group-split acc     : {acc_grp:.4f}  (drop {acc_rand-acc_grp:+.4f})")
    print(f"  best LLM acc        : 0.477 (unaffected: zero/few-shot, not trained)")
    print(f"  gap random -> group : {acc_rand-0.477:.3f} -> {acc_grp-0.477:.3f}")


if __name__ == "__main__":
    main()
