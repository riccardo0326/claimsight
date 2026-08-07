"""Hugging Face LayoutLM document-question-answering extractor.

Uses word_boxes from the PDF text layer so Tesseract OCR is never required.
LayoutLM-v1 does not need pixel_values, so image=None is safe.
"""

from __future__ import annotations

import logging
from typing import Any

from api.config import get_settings
from agents.extractors.pdf_text import WordBox

logger = logging.getLogger(__name__)

_pipeline: Any | None = None


def _get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline

        settings = get_settings()
        logger.info("Loading DocVQA model %s", settings.doc_qa_model)
        _pipeline = pipeline(
            "document-question-answering",
            model=settings.doc_qa_model,
        )
    return _pipeline


class LayoutLMExtractor:
    """Lazy-loading LayoutLM DocVQA extractor (module-level pipeline singleton)."""

    def answer(self, word_boxes: list[WordBox], question: str) -> tuple[str | None, float]:
        if not word_boxes:
            return None, 0.0

        pipe = _get_pipeline()
        # Pass image=None + word_boxes to skip OCR entirely (LayoutLM-v1 path).
        results = pipe(image=None, question=question, word_boxes=word_boxes)
        if not results:
            return None, 0.0

        # Pipeline may return a dict or a list of dicts depending on version / top_k.
        if isinstance(results, list):
            best = results[0] if results else {}
        else:
            best = results

        answer = best.get("answer")
        score = float(best.get("score") or 0.0)
        if answer is None or str(answer).strip() == "":
            return None, score
        return str(answer).strip(), score


def get_layoutlm_extractor() -> LayoutLMExtractor:
    return LayoutLMExtractor()
