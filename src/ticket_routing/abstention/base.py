"""Abstention policy interface.

A policy maps (predictions, confidences, agreement votes if applicable) to one
of two decisions per item: AUTO_ROUTE or DEFER_TO_HUMAN.
"""
from __future__ import annotations

import abc
from typing import List, Optional, Sequence

from ..models.base import PredictionBatch


AUTO_ROUTE = "AUTO_ROUTE"
DEFER_TO_HUMAN = "DEFER_TO_HUMAN"


class AbstentionPolicy(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def decide(
        self,
        primary: PredictionBatch,
        confidences: Sequence[float],
        auxiliary: Optional[Sequence[PredictionBatch]] = None,
    ) -> List[str]: ...
