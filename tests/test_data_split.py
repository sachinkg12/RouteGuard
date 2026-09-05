"""Stratified split preserves labels; train/calib/test are disjoint; few-shot
examples come from train only."""
from __future__ import annotations

from collections import Counter

import pytest

from ticket_routing.data.loaders import SyntheticTicketLoader
from ticket_routing.data.splitters import stratified_three_way_split
from ticket_routing.models.llm_prompt_classifier import LLMPromptClassifier
from ticket_routing.llm.mock_client import DryRunMockClient


def test_stratified_split_preserves_labels():
    bundle = SyntheticTicketLoader(n_per_class=30, seed=0).load()
    split = stratified_three_way_split(
        bundle, train_fraction=0.7, calibration_fraction=0.1, test_fraction=0.2, seed=0
    )
    full = Counter(bundle.labels)
    train_c = Counter(split.train_labels)
    test_c = Counter(split.test_labels)
    for lbl in full:
        # Each label appears in train and test under any reasonable split.
        assert train_c[lbl] >= 1
        assert test_c[lbl] >= 1
    assert sum(full.values()) == sum(train_c.values()) + sum(Counter(split.calib_labels).values()) + sum(test_c.values())


def test_no_leakage_between_splits():
    bundle = SyntheticTicketLoader(n_per_class=20, seed=1).load()
    split = stratified_three_way_split(
        bundle, train_fraction=0.7, calibration_fraction=0.1, test_fraction=0.2, seed=1
    )
    train = set(split.train_texts)
    calib = set(split.calib_texts)
    test = set(split.test_texts)
    assert train & calib == set()
    assert train & test == set()
    assert calib & test == set()


def test_few_shot_examples_come_only_from_train():
    bundle = SyntheticTicketLoader(n_per_class=20, seed=2).load()
    split = stratified_three_way_split(
        bundle, 0.7, 0.1, 0.2, seed=2
    )
    classifier = LLMPromptClassifier(
        client=DryRunMockClient(),
        setting="few_shot",
        k_shot_per_class=3,
        random_state=2,
    )
    classifier.fit(split.train_texts, split.train_labels, split.label_names)
    chosen_texts = {ex_text for ex_text, _ in classifier._few_shot_examples}
    assert chosen_texts.issubset(set(split.train_texts))
    assert chosen_texts.isdisjoint(set(split.calib_texts))
    assert chosen_texts.isdisjoint(set(split.test_texts))


def test_split_fractions_must_sum_to_one():
    bundle = SyntheticTicketLoader(n_per_class=10, seed=3).load()
    with pytest.raises(ValueError):
        stratified_three_way_split(bundle, 0.5, 0.1, 0.2, seed=3)
