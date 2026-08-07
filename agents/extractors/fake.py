"""Deterministic DocVQA fake for unit/integration tests (no HF model load)."""

from __future__ import annotations

import re
from typing import Mapping

from agents.extractors.pdf_text import WordBox


class FakeDocVQAExtractor:
    """Maps known questions to canned (answer, score) pairs.

    Falls back to a simple keyword scan over word_boxes when a question is
    not in the canned map — useful for negative-path tests that inject
    low scores via the canned map.
    """

    def __init__(
        self,
        answers: Mapping[str, tuple[str | None, float]] | None = None,
        default_score: float = 0.95,
    ) -> None:
        self._answers = {k.lower(): v for k, v in (answers or {}).items()}
        self.default_score = default_score

    def answer(self, word_boxes: list[WordBox], question: str) -> tuple[str | None, float]:
        key = question.lower().strip()
        if key in self._answers:
            return self._answers[key]

        # Loose alias matching so slight question wording differences still hit.
        for canned_q, value in self._answers.items():
            if canned_q in key or key in canned_q:
                return value

        # Last-resort: return joined words so callers still get something.
        text = " ".join(w for w, _ in word_boxes)
        return (text[:80] if text else None), self.default_score


def fake_from_expected(expected: dict, score: float = 0.95) -> FakeDocVQAExtractor:
    """Build a fake extractor that returns values from fixtures/expected.json."""
    limits = expected.get("coverage_limits", {})
    answers = {
        "what is the policy id?": (expected.get("policy_id"), score),
        "what is the policy number?": (expected.get("policy_id"), score),
        "what is the deductible?": (f"${expected['deductible']:,.2f}", score)
        if expected.get("deductible") is not None
        else (None, 0.0),
        "what is the vin?": (expected.get("vin"), score),
        "what is the vehicle identification number?": (expected.get("vin"), score),
        "what is the incident date?": (expected.get("incident_date"), score),
        "what is the collision coverage limit?": (
            f"${limits['collision']:,.0f}" if "collision" in limits else None,
            score if "collision" in limits else 0.0,
        ),
        "what is the comprehensive coverage limit?": (
            f"${limits['comprehensive']:,.0f}" if "comprehensive" in limits else None,
            score if "comprehensive" in limits else 0.0,
        ),
        "what is the liability coverage limit?": (
            f"${limits['liability']:,.0f}" if "liability" in limits else None,
            score if "liability" in limits else 0.0,
        ),
    }
    return FakeDocVQAExtractor(answers=answers, default_score=score)


# Re-export regex helper used by tests that want to inspect question keys.
QUESTION_NORMALIZE = re.compile(r"\s+")
