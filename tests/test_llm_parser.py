"""LLM parser: malformed JSON, out-of-set labels, and valid outputs."""
from __future__ import annotations

from ticket_routing.llm.parser import parse_label_json
from ticket_routing.models.base import INVALID_LABEL, PARSE_ERROR, PARSE_OK


ALLOWED = ["network", "email", "hardware"]


def test_valid_json_in_set():
    out = parse_label_json('{"label": "network", "confidence": 0.9}', ALLOWED)
    assert out.status == PARSE_OK
    assert out.label == "network"
    assert out.confidence == 0.9


def test_label_out_of_set_flagged_invalid():
    out = parse_label_json('{"label": "rogue_label", "confidence": 0.5}', ALLOWED)
    assert out.status == INVALID_LABEL


def test_malformed_json_flagged_parse_error():
    out = parse_label_json("not json at all", ALLOWED)
    assert out.status == PARSE_ERROR


def test_confidence_clamps_to_unit_interval():
    out = parse_label_json('{"label": "email", "confidence": 1.5}', ALLOWED)
    assert out.confidence == 1.0
    out = parse_label_json('{"label": "email", "confidence": -0.2}', ALLOWED)
    assert out.confidence == 0.0


def test_embedded_json_block_is_extracted():
    text = "Sure: {\"label\": \"hardware\", \"confidence\": 0.75}\nlet me know."
    out = parse_label_json(text, ALLOWED)
    assert out.status == PARSE_OK
    assert out.label == "hardware"


def test_missing_label_is_parse_error():
    out = parse_label_json('{"confidence": 0.5}', ALLOWED)
    assert out.status == PARSE_ERROR


def test_json_after_brace_containing_preamble_is_extracted():
    # Regression: the old greedy `{.*}` regex failed when a reasoning preamble
    # contained braces (set notation, code) before the real JSON object.
    text = '<think>the set is {a, b}</think>\n{"label": "email", "confidence": 0.8}'
    out = parse_label_json(text, ALLOWED)
    assert out.status == PARSE_OK
    assert out.label == "email"
    assert out.confidence == 0.8


def test_first_valid_json_object_wins_over_trailing_brace():
    text = '{"label": "network", "confidence": 0.9}\nNote: use {curly}.'
    out = parse_label_json(text, ALLOWED)
    assert out.status == PARSE_OK
    assert out.label == "network"


def test_nested_json_object_parses():
    text = '{"label": "hardware", "confidence": 0.7, "meta": {"a": 1}}'
    out = parse_label_json(text, ALLOWED)
    assert out.status == PARSE_OK
    assert out.label == "hardware"
