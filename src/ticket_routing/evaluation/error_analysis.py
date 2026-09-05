"""Top-confused-pairs error analysis and representative examples per pair."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from ..models.base import PredictionBatch


def top_confused_pairs(
    batch: PredictionBatch,
    true_labels: Sequence[str],
    texts: Sequence[str],
    k: int = 5,
) -> List[Dict]:
    pairs: Counter = Counter()
    pair_examples: Dict[tuple[str, str], List[str]] = {}
    for text, pred, truth in zip(texts, batch.predicted_labels, true_labels):
        if pred == truth:
            continue
        key = (truth, pred)
        pairs[key] += 1
        pair_examples.setdefault(key, []).append(text)
    out: List[Dict] = []
    for (truth, pred), count in pairs.most_common(k):
        out.append(
            {
                "true_label": truth,
                "predicted_label": pred,
                "count": int(count),
                "example_ticket": pair_examples[(truth, pred)][0],
            }
        )
    return out
