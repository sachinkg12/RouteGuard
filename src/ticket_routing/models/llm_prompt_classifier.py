"""LLM-prompted classifier: prompt builder + LLM client + label-constrained parser.

Few-shot examples come exclusively from the train split (leakage guard enforced
in `fit()`). The predictor never sees test labels.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import List, Optional, Sequence

from .base import (
    INVALID_LABEL,
    PARSE_ERROR,
    PARSE_OK,
    PredictionBatch,
    Predictor,
)
from .registry import register
from ..llm.base import LLMClient, LLMResponse
from ..llm.parser import parse_label_json
from ..llm.prompt_templates import build_classification_prompt


@register("llm_prompt_classifier")
class LLMPromptClassifier(Predictor):
    name = "llm_prompt_classifier"

    def __init__(
        self,
        client: LLMClient,
        setting: str = "zero_shot",  # "zero_shot" or "few_shot"
        k_shot_per_class: int = 3,
        prompt_template_version: str = "v1",
        self_consistency_samples: int = 1,
        random_state: int = 42,
    ) -> None:
        if setting not in ("zero_shot", "few_shot"):
            raise ValueError(f"setting must be zero_shot or few_shot, got {setting}")
        self.client = client
        self.setting = setting
        self.k_shot_per_class = k_shot_per_class
        self.prompt_template_version = prompt_template_version
        self.self_consistency_samples = max(1, self_consistency_samples)
        self.random_state = random_state
        self._label_names: List[str] = []
        self._few_shot_examples: List[tuple[str, str]] = []
        # Stash full per-sample outputs so SelfConsistency confidence can read them.
        self._last_samples: List[List[dict]] = []

    # ----- fit: select few-shot exemplars (train-only). -----------------------

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[str],
        label_names: Sequence[str],
    ) -> None:
        self._label_names = list(label_names)
        if self.setting != "few_shot":
            self._few_shot_examples = []
            return

        import random

        rng = random.Random(self.random_state)
        by_label: dict[str, List[str]] = defaultdict(list)
        for t, lbl in zip(train_texts, train_labels):
            if lbl in self._label_names:
                by_label[lbl].append(t)

        exemplars: List[tuple[str, str]] = []
        for lbl in self._label_names:
            pool = by_label.get(lbl, [])
            rng.shuffle(pool)
            for ex in pool[: self.k_shot_per_class]:
                exemplars.append((ex, lbl))
        rng.shuffle(exemplars)
        self._few_shot_examples = exemplars

    # ----- predict: build prompt, run LLM, parse, optionally self-consistency. -

    def predict(self, texts: Sequence[str]) -> PredictionBatch:
        if not self._label_names:
            raise RuntimeError("LLMPromptClassifier.predict() called before fit().")

        predicted_labels: List[str] = []
        confidences: List[float] = []
        raw_outputs: List[str] = []
        parse_statuses: List[str] = []
        latencies: List[float] = []
        costs: List[float] = []
        per_sample_dump: List[List[dict]] = []

        for text in texts:
            prompt = build_classification_prompt(
                text=text,
                label_names=self._label_names,
                few_shot=self._few_shot_examples if self.setting == "few_shot" else None,
                version=self.prompt_template_version,
            )
            sample_records: List[dict] = []
            t0 = time.perf_counter()
            for _ in range(self.self_consistency_samples):
                # Persistent provider failures (SDK retries exhausted) become a
                # PARSE_ERROR row for THIS example rather than crashing the
                # whole multi-hour run.
                try:
                    response: LLMResponse = self.client.generate(prompt)
                except Exception as exc:  # pragma: no cover - network
                    sample_records.append(
                        {
                            "label": None,
                            "confidence": None,
                            "status": PARSE_ERROR,
                            "raw": f"__CLIENT_ERROR__: {type(exc).__name__}: {exc}",
                            "latency": time.perf_counter() - t0,
                            "cost": 0.0,
                        }
                    )
                    continue
                parsed = parse_label_json(response.text, self._label_names)
                sample_records.append(
                    {
                        "label": parsed.label,
                        "confidence": parsed.confidence,
                        "status": parsed.status,
                        "raw": response.text,
                        "latency": response.latency_seconds,
                        "cost": response.cost_estimate,
                    }
                )
            latency = time.perf_counter() - t0

            aggregated_label, aggregated_conf, aggregated_status = _aggregate_samples(
                sample_records, self._label_names
            )
            predicted_labels.append(aggregated_label)
            confidences.append(aggregated_conf)
            raw_outputs.append(sample_records[0]["raw"])
            parse_statuses.append(aggregated_status)
            latencies.append(latency)
            costs.append(sum(r.get("cost", 0.0) or 0.0 for r in sample_records))
            per_sample_dump.append(sample_records)

        self._last_samples = per_sample_dump
        return PredictionBatch(
            predicted_labels=predicted_labels,
            confidence_scores=confidences,
            raw_outputs=raw_outputs,
            parse_status=parse_statuses,
            latency_seconds=latencies,
            cost_estimate=costs,
            model_metadata={
                "predictor": self.name,
                "setting": self.setting,
                "k_shot_per_class": self.k_shot_per_class,
                "prompt_template_version": self.prompt_template_version,
                "self_consistency_samples": self.self_consistency_samples,
                "label_names": self._label_names,
                "client_kind": getattr(self.client, "kind", "unknown"),
                "client_model": getattr(self.client, "model", "unknown"),
            },
        )


def _aggregate_samples(
    samples: List[dict],
    label_names: List[str],
) -> tuple[str, float, str]:
    """Vote across self-consistency samples.

    - If any valid label appears, return the most-voted label, confidence = vote share.
    - Otherwise return the most common error status; predicted label = sentinel.
    """
    valid = [s for s in samples if s["status"] == PARSE_OK and s["label"] in label_names]
    if valid:
        votes: dict[str, int] = defaultdict(int)
        for s in valid:
            votes[s["label"]] += 1
        winner = max(votes.items(), key=lambda kv: kv[1])
        share = winner[1] / len(samples)
        # If only one sample, fall back to model-reported confidence when present.
        if len(samples) == 1 and valid[0]["confidence"] is not None:
            return valid[0]["label"], float(valid[0]["confidence"]), PARSE_OK
        return winner[0], float(share), PARSE_OK

    statuses = [s["status"] for s in samples]
    if any(s == INVALID_LABEL for s in statuses):
        return "__INVALID_LABEL__", 0.0, INVALID_LABEL
    return "__PARSE_ERROR__", 0.0, PARSE_ERROR
