"""Dataset loaders.

Each loader is a small class implementing `DatasetLoader.load() -> DatasetBundle`.
Adding a new dataset = adding a new loader; no other module changes.
"""
from __future__ import annotations

import abc
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .schema import DatasetBundle
from ..utils.hashing import sha256_iter_strings, sha256_obj


class DatasetLoader(abc.ABC):
    """Interface for dataset loaders."""

    @abc.abstractmethod
    def load(self) -> DatasetBundle: ...


class CsvTicketLoader(DatasetLoader):
    """Loads any public CSV with configurable column names.

    Used for Kaggle 'Customer Support Tickets', 'Consumer Complaints', etc.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | Path,
        text_column: str,
        label_column: str,
        minimum_class_count: int = 10,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = Path(dataset_path)
        self.text_column = text_column
        self.label_column = label_column
        self.minimum_class_count = minimum_class_count

    def load(self) -> DatasetBundle:
        df = pd.read_csv(self.dataset_path)
        if self.text_column not in df.columns or self.label_column not in df.columns:
            raise ValueError(
                f"CSV missing required columns. Found: {list(df.columns)}; "
                f"need text='{self.text_column}', label='{self.label_column}'."
            )
        df = df[[self.text_column, self.label_column]].dropna()
        df[self.text_column] = df[self.text_column].astype(str).str.strip()
        df[self.label_column] = df[self.label_column].astype(str).str.strip()
        df = df[df[self.text_column].str.len() > 0]

        counts = Counter(df[self.label_column].tolist())
        keep = {lbl for lbl, c in counts.items() if c >= self.minimum_class_count}
        df = df[df[self.label_column].isin(keep)].reset_index(drop=True)

        texts = df[self.text_column].tolist()
        labels = df[self.label_column].tolist()
        label_names = sorted(set(labels))
        dataset_hash = sha256_iter_strings(
            f"{t}\t{lbl}" for t, lbl in zip(texts, labels)
        )
        return DatasetBundle(
            texts=texts,
            labels=labels,
            label_names=label_names,
            metadata={
                "dataset_name": self.dataset_name,
                "source": str(self.dataset_path),
                "n_examples": len(texts),
                "n_classes": len(label_names),
            },
            dataset_hash=dataset_hash,
        )


class SyntheticTicketLoader(DatasetLoader):
    """Deterministic synthetic IT tickets used for tests / debug runs.

    Public surrogate: vocabulary and labels are generic IT triage categories
    that overlap with public ticket datasets (network, account, hardware...).
    """

    DEFAULT_CATEGORIES: dict[str, List[str]] = {
        "network": [
            "vpn keeps disconnecting from corporate network",
            "wifi unable to authenticate after password reset",
            "cannot reach internal subnet from remote office",
            "dns resolution failing for internal hostnames",
            "intermittent packet loss on the office switch",
        ],
        "account_access": [
            "user locked out after multiple failed sso attempts",
            "request to reset password for shared mailbox",
            "mfa device lost please re-enroll my account",
            "new starter needs access to the finance group",
            "permission denied opening shared drive folder",
        ],
        "hardware": [
            "laptop battery swelling please replace the unit",
            "monitor flickering when docked at workstation",
            "keyboard keys sticking after coffee spill",
            "printer jam on third floor copier device",
            "ssd making clicking noises and slow boot",
        ],
        "software_install": [
            "need adobe creative suite installed on new laptop",
            "request approval to install postgres on workstation",
            "outlook plugin missing after windows feature update",
            "please push the latest zoom client to my machine",
            "vpn client uninstalled itself after reboot",
        ],
        "email": [
            "outlook stuck on loading profile every morning",
            "calendar invites not syncing across devices",
            "shared mailbox stopped receiving external email",
            "spam filter quarantining legitimate vendor messages",
            "out of office reply not firing for external senders",
        ],
        "security_incident": [
            "received suspicious phishing email asking for credentials",
            "device flagged by edr for unknown executable",
            "unusual sign in from foreign ip address detected",
            "suspect malware after opening pdf attachment",
            "leaked api key in public repository please rotate",
        ],
    }

    def __init__(
        self,
        n_per_class: int = 80,
        seed: int = 42,
        categories: Optional[dict[str, List[str]]] = None,
    ) -> None:
        self.n_per_class = n_per_class
        self.seed = seed
        self.categories = categories or self.DEFAULT_CATEGORIES

    def load(self) -> DatasetBundle:
        rng = random.Random(self.seed)
        texts: List[str] = []
        labels: List[str] = []

        adjuncts = [
            "urgent",
            "please advise",
            "happening since yesterday",
            "intermittent issue",
            "affecting multiple users",
            "blocking work",
            "low priority",
            "after recent update",
            "remote worker",
            "office location",
        ]

        for label, seeds_ in self.categories.items():
            for i in range(self.n_per_class):
                base = rng.choice(seeds_)
                adj = rng.choice(adjuncts)
                noise = rng.choice(["", " thanks", " regards", " let me know"])
                text = f"{base} - {adj}{noise} (#{i:04d})"
                texts.append(text)
                labels.append(label)

        # Shuffle deterministically so loaders don't return labels in blocks.
        order = list(range(len(texts)))
        rng.shuffle(order)
        texts = [texts[i] for i in order]
        labels = [labels[i] for i in order]
        label_names = sorted(self.categories.keys())
        dataset_hash = sha256_obj(
            {
                "categories": sorted(self.categories.keys()),
                "n_per_class": self.n_per_class,
                "seed": self.seed,
            }
        )
        return DatasetBundle(
            texts=texts,
            labels=labels,
            label_names=label_names,
            metadata={
                "dataset_name": "synthetic_it_tickets",
                "source": "generated",
                "n_examples": len(texts),
                "n_classes": len(label_names),
            },
            dataset_hash=dataset_hash,
        )


def build_loader_from_config(dataset_cfg) -> DatasetLoader:
    """Factory dispatch — adding a loader requires extending this switch only."""
    name = dataset_cfg.dataset_name.lower()
    if name == "synthetic":
        return SyntheticTicketLoader(
            n_per_class=dataset_cfg.synthetic_n_per_class,
            seed=dataset_cfg.random_seed,
        )
    if dataset_cfg.dataset_path:
        return CsvTicketLoader(
            dataset_name=dataset_cfg.dataset_name,
            dataset_path=dataset_cfg.dataset_path,
            text_column=dataset_cfg.text_column,
            label_column=dataset_cfg.label_column,
            minimum_class_count=dataset_cfg.minimum_class_count,
        )
    raise ValueError(
        "Dataset config must either set dataset_name='synthetic' or provide dataset_path."
    )
