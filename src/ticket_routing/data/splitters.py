"""Stratified train / calibration / test splitter with leakage guards."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Tuple

import numpy as np

from .schema import DatasetBundle, SplitBundle


def stratified_three_way_split(
    bundle: DatasetBundle,
    train_fraction: float,
    calibration_fraction: float,
    test_fraction: float,
    seed: int,
) -> SplitBundle:
    total = train_fraction + calibration_fraction + test_fraction
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"split fractions must sum to 1.0, got {total} "
            f"(train={train_fraction}, calib={calibration_fraction}, test={test_fraction})"
        )

    rng = np.random.default_rng(seed)
    indices_by_label: dict[str, List[int]] = defaultdict(list)
    for i, lbl in enumerate(bundle.labels):
        indices_by_label[lbl].append(i)

    train_idx: List[int] = []
    calib_idx: List[int] = []
    test_idx: List[int] = []

    for lbl, idxs in indices_by_label.items():
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * train_fraction))
        n_calib = int(round(n * calibration_fraction))
        # Anything remaining goes to test, which keeps test_fraction >= configured.
        n_train = min(n_train, max(n - 2, 1))
        n_calib = min(n_calib, max(n - n_train - 1, 0))
        train_idx.extend(shuffled[:n_train])
        calib_idx.extend(shuffled[n_train : n_train + n_calib])
        test_idx.extend(shuffled[n_train + n_calib :])

    # Final shuffle (still seeded) so consumers can't accidentally rely on class order.
    for buf in (train_idx, calib_idx, test_idx):
        rng.shuffle(buf)

    _assert_disjoint(train_idx, calib_idx, test_idx)
    return SplitBundle(
        train_texts=[bundle.texts[i] for i in train_idx],
        train_labels=[bundle.labels[i] for i in train_idx],
        calib_texts=[bundle.texts[i] for i in calib_idx],
        calib_labels=[bundle.labels[i] for i in calib_idx],
        test_texts=[bundle.texts[i] for i in test_idx],
        test_labels=[bundle.labels[i] for i in test_idx],
        label_names=bundle.label_names,
        metadata={
            "split_seed": seed,
            "train_fraction": train_fraction,
            "calibration_fraction": calibration_fraction,
            "test_fraction": test_fraction,
            "n_train": len(train_idx),
            "n_calib": len(calib_idx),
            "n_test": len(test_idx),
        },
    )


def stratified_test_subsample_for_llm(
    test_texts: List[str],
    test_labels: List[str],
    max_total: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Cost-bounded stratified subsample of the test split for LLM scoring.

    Keeps class proportions, returns the full test set if `max_total` would exceed it.
    """
    if max_total >= len(test_texts):
        return test_texts, test_labels
    rng = np.random.default_rng(seed)
    indices_by_label: dict[str, List[int]] = defaultdict(list)
    for i, lbl in enumerate(test_labels):
        indices_by_label[lbl].append(i)

    total = len(test_texts)
    selected: List[int] = []
    for lbl, idxs in indices_by_label.items():
        # Floor allocation; remainder distributed in second pass below.
        quota = max(1, int(max_total * (len(idxs) / total)))
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        selected.extend(shuffled[:quota])

    # Trim or pad to exactly max_total without duplicating examples.
    if len(selected) > max_total:
        rng.shuffle(selected)
        selected = selected[:max_total]
    elif len(selected) < max_total:
        remaining = [i for i in range(total) if i not in set(selected)]
        rng.shuffle(remaining)
        deficit = max_total - len(selected)
        selected.extend(remaining[:deficit])

    rng.shuffle(selected)
    return (
        [test_texts[i] for i in selected],
        [test_labels[i] for i in selected],
    )


def _assert_disjoint(*groups: Iterable[int]) -> None:
    seen: set[int] = set()
    for g in groups:
        for i in g:
            if i in seen:
                raise AssertionError(
                    "Leakage detected: index appears in more than one split."
                )
            seen.add(i)
