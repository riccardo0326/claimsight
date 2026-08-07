"""Document Agent — extract structured fields from policy + estimate PDFs.

Contract: PROJECT_SPEC.md §6.2 DocumentOutput.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from dateutil import parser as date_parser

from agents.extractors.base import DocVQAExtractor
from agents.extractors.layoutlm import get_layoutlm_extractor
from agents.extractors.pdf_text import WordBox, extract_line_items, extract_word_boxes
from agents.schemas import DocumentOutput
from api.config import get_settings

logger = logging.getLogger(__name__)

VIN_RE = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b", re.IGNORECASE)
CURRENCY_RE = re.compile(r"[^0-9.\-]")

# Targeted DocVQA questions against the policy PDF.
POLICY_ID_QUESTIONS = (
    "What is the policy id?",
    "What is the policy number?",
)
DEDUCTIBLE_QUESTION = "What is the deductible?"
VIN_QUESTIONS = (
    "What is the VIN?",
    "What is the vehicle identification number?",
)
INCIDENT_DATE_QUESTION = "What is the incident date?"
COVERAGE_QUESTIONS = {
    "collision": "What is the collision coverage limit?",
    "comprehensive": "What is the comprehensive coverage limit?",
    "liability": "What is the liability coverage limit?",
}


def _parse_currency(raw: str | None) -> float | None:
    if raw is None:
        return None
    cleaned = CURRENCY_RE.sub("", raw.strip())
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_vin(raw: str | None) -> str | None:
    if raw is None:
        return None
    match = VIN_RE.search(raw.upper().replace(" ", ""))
    if match:
        return match.group(1).upper()
    # Also accept an exact 17-char candidate after stripping punctuation.
    candidate = re.sub(r"[^A-HJ-NPR-Z0-9]", "", raw.upper())
    if len(candidate) == 17 and VIN_RE.fullmatch(candidate):
        return candidate
    return None


def _parse_date(raw: str | None) -> date | None:
    if raw is None:
        return None
    try:
        parsed = date_parser.parse(raw, fuzzy=True)
        if isinstance(parsed, datetime):
            return parsed.date()
        return parsed
    except (ValueError, OverflowError, TypeError):
        return None


def _best_answer(
    extractor: DocVQAExtractor,
    pages: list[list[WordBox]],
    questions: tuple[str, ...] | str,
    min_confidence: float,
    field_name: str,
    confidences: dict[str, float],
    low_confidence_fields: list[str],
) -> str | None:
    q_list = (questions,) if isinstance(questions, str) else questions
    best_text: str | None = None
    best_score = -1.0

    for page_boxes in pages:
        for question in q_list:
            text, score = extractor.answer(page_boxes, question)
            if score > best_score:
                best_score = score
                best_text = text

    confidences[field_name] = max(best_score, 0.0)
    if best_score < min_confidence or best_text is None:
        logger.info(
            "DocVQA miss field=%s score=%.4f threshold=%.4f answer=%r",
            field_name,
            best_score,
            min_confidence,
            best_text,
        )
        low_confidence_fields.append(field_name)
        return None
    return best_text


def run_document_agent(
    policy_pdf_path: str | Path,
    estimate_pdf_path: str | Path,
    *,
    extractor: DocVQAExtractor | None = None,
    min_confidence: float | None = None,
) -> tuple[DocumentOutput, dict[str, Any]]:
    """Run Document Agent and return (DocumentOutput, extraction_meta)."""
    settings = get_settings()
    threshold = settings.doc_qa_min_confidence if min_confidence is None else min_confidence
    qa = extractor if extractor is not None else get_layoutlm_extractor()

    pages = extract_word_boxes(policy_pdf_path)
    confidences: dict[str, float] = {}
    low_confidence_fields: list[str] = []

    policy_raw = _best_answer(
        qa, pages, POLICY_ID_QUESTIONS, threshold, "policy_id", confidences, low_confidence_fields
    )
    # Normalize policy id: keep alnum + hyphen tokens that look like IDs.
    policy_id = None
    if policy_raw:
        match = re.search(r"(POL-[\w\-]+|[\w\-]{6,})", policy_raw, re.IGNORECASE)
        policy_id = match.group(1) if match else policy_raw.strip()

    deductible_raw = _best_answer(
        qa, pages, DEDUCTIBLE_QUESTION, threshold, "deductible", confidences, low_confidence_fields
    )
    deductible = _parse_currency(deductible_raw)
    if deductible_raw is not None and deductible is None:
        logger.info("DocVQA unparseable currency for deductible: %r", deductible_raw)
        if "deductible" not in low_confidence_fields:
            low_confidence_fields.append("deductible")

    vin_raw = _best_answer(
        qa, pages, VIN_QUESTIONS, threshold, "vin", confidences, low_confidence_fields
    )
    vin = _parse_vin(vin_raw)
    if vin_raw is not None and vin is None:
        logger.info("DocVQA invalid VIN rejected: %r", vin_raw)
        if "vin" not in low_confidence_fields:
            low_confidence_fields.append("vin")

    date_raw = _best_answer(
        qa,
        pages,
        INCIDENT_DATE_QUESTION,
        threshold,
        "incident_date",
        confidences,
        low_confidence_fields,
    )
    incident_date = _parse_date(date_raw)
    if date_raw is not None and incident_date is None:
        logger.info("DocVQA unparseable date rejected: %r", date_raw)
        if "incident_date" not in low_confidence_fields:
            low_confidence_fields.append("incident_date")

    coverage_limits: dict[str, float] = {}
    for key, question in COVERAGE_QUESTIONS.items():
        field = f"coverage_limits.{key}"
        raw = _best_answer(
            qa, pages, question, threshold, field, confidences, low_confidence_fields
        )
        value = _parse_currency(raw)
        if value is not None:
            coverage_limits[key] = value
        elif raw is not None:
            logger.info("DocVQA unparseable coverage limit %s: %r", key, raw)

    # Line items from estimate PDF via pdfplumber table extraction (placeholder for TableQA).
    line_items = extract_line_items(estimate_pdf_path)

    output = DocumentOutput(
        policy_id=policy_id,
        coverage_limits=coverage_limits,
        deductible=deductible,
        vin=vin,
        incident_date=incident_date,
        line_items=line_items,
    )
    meta = {
        "confidences": confidences,
        "low_confidence_fields": low_confidence_fields,
        "min_confidence": threshold,
    }
    return output, meta
