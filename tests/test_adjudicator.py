"""Mocked-LLM Adjudicator tests (no network / no real frontier API)."""

from __future__ import annotations

import json

from agents.adjudicator import run_adjudicator
from agents.schemas import (
    ClaimReport,
    DocumentOutput,
    RAGOutput,
    RetrievedClause,
    RiskFlag,
    RiskOutput,
    VerifierOutput,
    VisionOutput,
)


def _doc(**kwargs) -> DocumentOutput:
    base = dict(
        policy_id="POL-2024-0098213",
        coverage_limits={"collision": 50000.0},
        deductible=500.0,
        vin="1HGCM82633A004352",
    )
    base.update(kwargs)
    return DocumentOutput(**base)


def _rag(*ids: str) -> RAGOutput:
    return RAGOutput(
        retrieved_clauses=[
            RetrievedClause(clause_id=i, text=f"Coverage text {i}", similarity_score=0.85)
            for i in ids
        ]
    )


def _risk(flags: list[RiskFlag] | None = None, score: float = 0.0) -> RiskOutput:
    return RiskOutput(flags=flags or [], risk_score=score)


def _llm_json(payload: dict):
    def _complete(_messages):
        return json.dumps(payload)

    return _complete


def test_valid_approve_output():
    report = run_adjudicator(
        narrative="Front-end collision with another vehicle.",
        document=_doc(),
        extraction_meta={"low_confidence_fields": [], "confidences": {}},
        vision=VisionOutput(
            detections=[],
            severity_tier="moderate damage",
            severity_confidence=0.7,
            vqa_answers={},
            low_confidence=False,
        ),
        rag=_rag("COL-001", "COL-002"),
        verifiers=VerifierOutput(),
        risk=_risk(),
        llm_complete=_llm_json(
            {
                "decision": "approve",
                "confidence": 0.99,
                "cited_clauses": ["COL-001"],
                "risk_flags": [],
                "reasoning_summary": "FACT: COL-001 covers collision. INFERENCE: photos support damage.",
            }
        ),
    )
    assert report.decision == "approve"
    assert report.cited_clauses == ["COL-001"]
    # Deterministic confidence replaces LLM self-score.
    assert report.confidence != 0.99
    assert 0.0 <= report.confidence <= 1.0


def test_valid_deny_output():
    report = run_adjudicator(
        narrative="Intentional damage admitted.",
        document=_doc(),
        extraction_meta={"low_confidence_fields": []},
        vision=None,
        rag=_rag("EXC-001"),
        verifiers=VerifierOutput(),
        risk=_risk(),
        llm_complete=_llm_json(
            {
                "decision": "deny",
                "confidence": 0.8,
                "cited_clauses": ["EXC-001"],
                "risk_flags": [],
                "reasoning_summary": "FACT: EXC-001 excludes intentional acts.",
            }
        ),
    )
    assert report.decision == "deny"
    assert report.cited_clauses == ["EXC-001"]


def test_valid_needs_review_output():
    report = run_adjudicator(
        narrative="Unclear cause.",
        document=_doc(),
        extraction_meta={"low_confidence_fields": []},
        vision=None,
        rag=_rag("COL-001"),
        verifiers=VerifierOutput(),
        risk=_risk(),
        llm_complete=_llm_json(
            {
                "decision": "needs_review",
                "confidence": 0.4,
                "cited_clauses": [],
                "risk_flags": [],
                "reasoning_summary": "UNKNOWN: insufficient evidence to approve or deny.",
            }
        ),
    )
    assert report.decision == "needs_review"


def test_malformed_json_forced_to_review():
    report = run_adjudicator(
        narrative="x",
        document=_doc(),
        extraction_meta={},
        vision=None,
        rag=_rag("COL-001"),
        verifiers=VerifierOutput(),
        risk=_risk(),
        llm_complete=lambda _m: "not-json{{{",
    )
    assert report.decision == "needs_review"
    assert "schema/parse" in report.reasoning_summary or "validation" in report.reasoning_summary


def test_invalid_enum_forced_to_review():
    report = run_adjudicator(
        narrative="x",
        document=_doc(),
        extraction_meta={},
        vision=None,
        rag=_rag("COL-001"),
        verifiers=VerifierOutput(),
        risk=_risk(),
        llm_complete=_llm_json(
            {
                "decision": "maybe",
                "confidence": 0.5,
                "cited_clauses": ["COL-001"],
                "risk_flags": [],
                "reasoning_summary": "bad enum",
            }
        ),
    )
    assert report.decision == "needs_review"


def test_missing_required_field_forced_to_review():
    report = run_adjudicator(
        narrative="x",
        document=_doc(),
        extraction_meta={},
        vision=None,
        rag=_rag("COL-001"),
        verifiers=VerifierOutput(),
        risk=_risk(),
        llm_complete=_llm_json(
            {
                "decision": "approve",
                "confidence": 0.5,
                "cited_clauses": ["COL-001"],
                # missing reasoning_summary
            }
        ),
    )
    assert report.decision == "needs_review"


def test_unknown_citation_forced_to_review():
    report = run_adjudicator(
        narrative="Collision.",
        document=_doc(),
        extraction_meta={"low_confidence_fields": []},
        vision=None,
        rag=_rag("COL-001"),
        verifiers=VerifierOutput(),
        risk=_risk(),
        llm_complete=_llm_json(
            {
                "decision": "approve",
                "confidence": 0.9,
                "cited_clauses": ["NOT-IN-RAG"],
                "risk_flags": [],
                "reasoning_summary": "Invented citation.",
            }
        ),
    )
    assert report.decision == "needs_review"


def test_approve_with_empty_rag_forced_to_review():
    report = run_adjudicator(
        narrative="Collision.",
        document=_doc(policy_id=None),
        extraction_meta={"low_confidence_fields": ["policy_id"]},
        vision=None,
        rag=RAGOutput(retrieved_clauses=[]),
        verifiers=VerifierOutput(),
        risk=_risk(),
        llm_complete=_llm_json(
            {
                "decision": "approve",
                "confidence": 0.95,
                "cited_clauses": [],
                "risk_flags": [],
                "reasoning_summary": "Should not approve without RAG.",
            }
        ),
    )
    assert report.decision == "needs_review"


def test_model_cites_nonexistent_clause():
    report = run_adjudicator(
        narrative="Hail.",
        document=_doc(),
        extraction_meta={"low_confidence_fields": []},
        vision=None,
        rag=_rag("COM-001"),
        verifiers=VerifierOutput(),
        risk=_risk(),
        llm_complete=_llm_json(
            {
                "decision": "approve",
                "confidence": 0.9,
                "cited_clauses": ["COM-001", "HALLUCINATED-42"],
                "risk_flags": [],
                "reasoning_summary": "Bad cite mixed with good.",
            }
        ),
    )
    assert report.decision == "needs_review"
    assert "HALLUCINATED-42" not in report.cited_clauses


def test_high_confidence_approval_insufficient_evidence():
    report = run_adjudicator(
        narrative="Unclear.",
        document=_doc(),
        extraction_meta={"low_confidence_fields": ["policy_id"]},
        vision=None,
        rag=_rag("COL-001"),
        verifiers=VerifierOutput(),
        risk=_risk(),
        llm_complete=_llm_json(
            {
                "decision": "approve",
                "confidence": 1.0,
                "cited_clauses": ["COL-001"],
                "risk_flags": [],
                "reasoning_summary": "Overconfident with low-confidence policy_id.",
            }
        ),
    )
    assert report.decision == "needs_review"


def test_upstream_risk_flags_overwrite_llm_flags():
    upstream = [
        RiskFlag(flag_type="weather mismatch", rationale="no storm", severity="medium")
    ]
    report = run_adjudicator(
        narrative="Hail damage.",
        document=_doc(),
        extraction_meta={"low_confidence_fields": []},
        vision=None,
        rag=_rag("COM-001"),
        verifiers=VerifierOutput(),
        risk=_risk(flags=upstream, score=0.3),
        llm_complete=_llm_json(
            {
                "decision": "approve",
                "confidence": 0.9,
                "cited_clauses": ["COM-001"],
                "risk_flags": [
                    {"flag_type": "invented", "rationale": "x", "severity": "high"}
                ],
                "reasoning_summary": "Model invented flags and approved.",
            }
        ),
    )
    assert report.decision == "needs_review"  # material risk blocks approve
    assert report.risk_flags == upstream


def test_llm_error_forced_to_review():
    from agents.llm_openai import LLMError

    def boom(_messages):
        raise LLMError("no key")

    report = run_adjudicator(
        narrative="x",
        document=_doc(),
        extraction_meta={},
        vision=None,
        rag=_rag("COL-001"),
        verifiers=VerifierOutput(),
        risk=_risk(),
        llm_complete=boom,
    )
    assert report.decision == "needs_review"
    assert "LLM failure" in report.reasoning_summary
