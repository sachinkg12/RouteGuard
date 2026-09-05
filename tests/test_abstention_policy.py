"""Threshold + agreement abstention policies behave as specified."""
from __future__ import annotations

import pytest

from ticket_routing.abstention.agreement_policy import AgreementAbstentionPolicy
from ticket_routing.abstention.always_route import AlwaysRoutePolicy
from ticket_routing.abstention.threshold_policy import ThresholdAbstentionPolicy
from ticket_routing.models.base import (
    INVALID_LABEL,
    PARSE_ERROR,
    PARSE_OK,
    PredictionBatch,
)


def _b(preds, status=None):
    return PredictionBatch(predicted_labels=preds, parse_status=status or [PARSE_OK] * len(preds))


def test_threshold_policy_above_threshold_is_auto_route():
    pol = ThresholdAbstentionPolicy(threshold=0.7)
    batch = _b(["a", "b", "a"])
    decisions = pol.decide(batch, [0.9, 0.6, 0.7])
    assert decisions == ["AUTO_ROUTE", "DEFER_TO_HUMAN", "AUTO_ROUTE"]


def test_threshold_policy_defers_on_parse_failure():
    pol = ThresholdAbstentionPolicy(threshold=0.5)
    batch = _b(["a", "x", "a"], status=[PARSE_OK, PARSE_ERROR, INVALID_LABEL])
    decisions = pol.decide(batch, [0.99, 0.99, 0.99])
    assert decisions == ["AUTO_ROUTE", "DEFER_TO_HUMAN", "DEFER_TO_HUMAN"]


def test_threshold_must_be_in_range():
    with pytest.raises(ValueError):
        ThresholdAbstentionPolicy(threshold=1.5)


def test_agreement_policy_two_of_three():
    primary = _b(["a", "a", "b"])
    aux1 = _b(["a", "b", "b"])
    aux2 = _b(["c", "b", "b"])
    pol = AgreementAbstentionPolicy(min_agree=2)
    decisions = pol.decide(primary, [1.0, 1.0, 1.0], auxiliary=[aux1, aux2])
    # i=0: primary a, aux a, aux c -> agree=2 -> route
    # i=1: primary a, aux b, aux b -> agree=1 -> defer
    # i=2: primary b, aux b, aux b -> agree=3 -> route
    assert decisions == ["AUTO_ROUTE", "DEFER_TO_HUMAN", "AUTO_ROUTE"]


def test_always_route_policy_never_defers():
    pol = AlwaysRoutePolicy()
    batch = _b(["a", "b"])
    assert pol.decide(batch, [0.0, 0.0]) == ["AUTO_ROUTE", "AUTO_ROUTE"]
