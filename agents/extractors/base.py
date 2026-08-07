from __future__ import annotations

from typing import Protocol

from agents.extractors.pdf_text import WordBox


class DocVQAExtractor(Protocol):
    """Protocol for document-question-answering backends."""

    def answer(self, word_boxes: list[WordBox], question: str) -> tuple[str | None, float]:
        """Return (answer_text, confidence_score). answer_text is None on empty output."""
        ...
