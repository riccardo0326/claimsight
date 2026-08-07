"""Unit tests for the Document Agent using fixture PDFs + fake DocVQA."""

from __future__ import annotations

from datetime import date

import pytest

from agents.document_agent import run_document_agent
from agents.extractors.fake import FakeDocVQAExtractor, fake_from_expected
from agents.extractors.pdf_text import extract_line_items
from agents.schemas import DocumentOutput


def test_extract_line_items_from_fixture(estimate_pdf, expected):
    items = extract_line_items(estimate_pdf)
    assert len(items) == len(expected["line_items"])
    for got, want in zip(items, expected["line_items"], strict=True):
        assert got.description == want["description"]
        assert got.cost == pytest.approx(want["cost"])


def test_document_agent_extracts_all_fields(policy_pdf, estimate_pdf, expected, fake_extractor):
    output, meta = run_document_agent(
        policy_pdf, estimate_pdf, extractor=fake_extractor, min_confidence=0.5
    )
    assert isinstance(output, DocumentOutput)
    assert output.policy_id == expected["policy_id"]
    assert output.coverage_limits == expected["coverage_limits"]
    assert output.deductible == pytest.approx(expected["deductible"])
    assert output.vin == expected["vin"]
    assert output.incident_date == date.fromisoformat(expected["incident_date"])
    assert len(output.line_items) == len(expected["line_items"])
    for got, want in zip(output.line_items, expected["line_items"], strict=True):
        assert got.description == want["description"]
        assert got.cost == pytest.approx(want["cost"])
    assert meta["low_confidence_fields"] == []


def test_low_confidence_returns_none(policy_pdf, estimate_pdf, expected):
    answers = {
        "what is the policy id?": (expected["policy_id"], 0.1),
        "what is the policy number?": (expected["policy_id"], 0.1),
        "what is the deductible?": (f"${expected['deductible']:,.2f}", 0.95),
        "what is the vin?": (expected["vin"], 0.95),
        "what is the vehicle identification number?": (expected["vin"], 0.95),
        "what is the incident date?": (expected["incident_date"], 0.95),
        "what is the collision coverage limit?": ("$50,000", 0.95),
        "what is the comprehensive coverage limit?": ("$25,000", 0.95),
        "what is the liability coverage limit?": ("$100,000", 0.95),
    }
    extractor = FakeDocVQAExtractor(answers=answers)
    output, meta = run_document_agent(
        policy_pdf, estimate_pdf, extractor=extractor, min_confidence=0.5
    )
    assert output.policy_id is None
    assert "policy_id" in meta["low_confidence_fields"]
    assert output.deductible == pytest.approx(expected["deductible"])


def test_malformed_vin_rejected(policy_pdf, estimate_pdf, expected):
    base = fake_from_expected(expected)
    answers = dict(base._answers)
    answers["what is the vin?"] = ("NOT-A-VIN", 0.99)
    answers["what is the vehicle identification number?"] = ("NOT-A-VIN", 0.99)
    extractor = FakeDocVQAExtractor(answers=answers)
    output, meta = run_document_agent(
        policy_pdf, estimate_pdf, extractor=extractor, min_confidence=0.5
    )
    assert output.vin is None
    assert "vin" in meta["low_confidence_fields"]


def test_unparseable_date_rejected(policy_pdf, estimate_pdf, expected):
    base = fake_from_expected(expected)
    answers = dict(base._answers)
    answers["what is the incident date?"] = ("not-a-date-xxx", 0.99)
    extractor = FakeDocVQAExtractor(answers=answers)
    output, meta = run_document_agent(
        policy_pdf, estimate_pdf, extractor=extractor, min_confidence=0.5
    )
    assert output.incident_date is None
    assert "incident_date" in meta["low_confidence_fields"]


@pytest.mark.hf
def test_document_agent_real_model_loose(policy_pdf, estimate_pdf, expected):
    """Optional: load real LayoutLM and assert key fields are non-null / plausible."""
    output, meta = run_document_agent(policy_pdf, estimate_pdf)
    # Loose assertions — exact match is not guaranteed on synthetic PDFs.
    assert output.policy_id is not None or "policy_id" in meta["low_confidence_fields"]
    if output.vin is not None:
        assert len(output.vin) == 17
    if output.deductible is not None:
        assert output.deductible > 0
    assert len(output.line_items) == len(expected["line_items"])
