"""Shared dataset types — keep frozen so they can be passed between modules safely."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass(frozen=True)
class DatasetBundle:
    texts: List[str]
    labels: List[str]
    label_names: List[str]
    metadata: dict = field(default_factory=dict)
    dataset_hash: str = ""

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have the same length")

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class SplitBundle:
    train_texts: List[str]
    train_labels: List[str]
    calib_texts: List[str]
    calib_labels: List[str]
    test_texts: List[str]
    test_labels: List[str]
    label_names: List[str]
    metadata: dict = field(default_factory=dict)

    def subset_indices(self, split: str) -> Sequence[int]:
        # Provided for callers that need positional access; mostly informational.
        sizes = {"train": len(self.train_texts), "calib": len(self.calib_texts), "test": len(self.test_texts)}
        return range(sizes[split])
