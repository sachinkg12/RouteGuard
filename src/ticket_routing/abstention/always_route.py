"""Baseline policy: never defer."""
from __future__ import annotations

from typing import List, Optional, Sequence

from .base import AUTO_ROUTE, AbstentionPolicy
from ..models.base import PredictionBatch


class AlwaysRoutePolicy(AbstentionPolicy):
    name = "always_route"

    def decide(
        self,
        primary: PredictionBatch,
        confidences: Sequence[float],
        auxiliary: Optional[Sequence[PredictionBatch]] = None,
    ) -> List[str]:
        return [AUTO_ROUTE] * len(primary)
