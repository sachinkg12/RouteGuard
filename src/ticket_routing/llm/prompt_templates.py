"""Versioned prompt templates so result bundles can record which version was used."""
from __future__ import annotations

from typing import List, Optional, Sequence


PROMPT_V1 = """You are classifying IT support tickets.

Choose exactly one label from this list:
{labels}

{few_shot_block}Ticket:
{ticket_text}

Return JSON only:
{{
  "label": "<one allowed label>",
  "confidence": <number from 0 to 1>
}}
"""


# v2: strict-format variant. Trailing "JSON:" cue makes the model continue
# directly with the JSON object instead of emitting a conversational preamble.
PROMPT_V2 = """[FORMAT] Output exactly one JSON object: {{"label": "<label>", "confidence": <number>}}. No preamble. No explanation. No markdown. JSON only.

Classify the IT support ticket below. Use exactly one of these labels:
{labels}

{few_shot_block}Ticket: {ticket_text}
JSON:"""


_TEMPLATES = {"v1": PROMPT_V1, "v2": PROMPT_V2}


def build_classification_prompt(
    text: str,
    label_names: Sequence[str],
    few_shot: Optional[Sequence[tuple[str, str]]] = None,
    version: str = "v1",
) -> str:
    if version not in _TEMPLATES:
        raise ValueError(f"Unknown prompt template version: {version}")
    labels_str = ", ".join(label_names)
    few_shot_block = ""
    if few_shot:
        rows = []
        for ex_text, ex_label in few_shot:
            rows.append(
                f"Ticket: {ex_text}\n"
                f"JSON: {{\"label\": \"{ex_label}\", \"confidence\": 1.0}}"
            )
        few_shot_block = "Examples:\n" + "\n\n".join(rows) + "\n\n"
    return _TEMPLATES[version].format(
        labels=labels_str,
        few_shot_block=few_shot_block,
        ticket_text=text,
    )
