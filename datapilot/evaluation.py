"""Deterministic quality gates for evidence-grounded LLM narratives."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

EVIDENCE_ID = re.compile(r"\bEVD-[A-Z0-9-]+\b")
NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")
PII = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|(?:\+?\d[\d .()-]{7,}\d)", re.I)
INJECTION = re.compile(
    r"ignore (?:all |the )?(?:previous|prior) instructions|system prompt|developer message",
    re.I,
)


@dataclass(frozen=True)
class NarrativeEvaluation:
    supported_evidence_ids: bool
    unsupported_numbers: list[str]
    leaked_pii: bool
    prompt_injection_echo: bool
    score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_narrative(
    output: str,
    allowed_evidence_ids: Iterable[str],
    allowed_numbers: Iterable[str | float | int],
) -> NarrativeEvaluation:
    """Score an AI narrative for evidence citations, numeric faithfulness, and safety."""
    allowed_ids = set(allowed_evidence_ids)
    cited = set(EVIDENCE_ID.findall(output))
    supported_ids = cited.issubset(allowed_ids) and bool(cited)
    allowed_numeric = {_normalize_number(str(value)) for value in allowed_numbers}
    narrative_without_ids = EVIDENCE_ID.sub("", output)
    found = {_normalize_number(value) for value in NUMBER.findall(narrative_without_ids)}
    unsupported = sorted(value for value in found if value not in allowed_numeric)
    leaked_pii = bool(PII.search(output))
    injection = bool(INJECTION.search(output))
    penalties = (0 if supported_ids else 0.35) + min(0.35, len(unsupported) * 0.1)
    penalties += 0.2 if leaked_pii else 0
    penalties += 0.1 if injection else 0
    return NarrativeEvaluation(
        supported_evidence_ids=supported_ids,
        unsupported_numbers=unsupported,
        leaked_pii=leaked_pii,
        prompt_injection_echo=injection,
        score=round(max(0.0, 1.0 - penalties), 3),
    )


def _normalize_number(value: str) -> str:
    return value.replace(",", "").rstrip("%").lstrip("+")
