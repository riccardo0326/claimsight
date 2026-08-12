"""Deterministic Adjudicator guardrail tests (no LLM / no network)."""

from __future__ import annotations

from agents.adjudicator_guardrails import apply_guardrails, compute_confidence
from agents.schemas import (
    ClaimReport,
    Detection,
    RAGOutput,
    RetrievedClause,
    RiskFlag,
    RiskOutput,
    VerifierOutput,
    VisionOutput,
    WeatherAtIncident,
)


def _clause(cid: str = "COL-001") -> RetrievedClause:
    return RetrievedClause(clause_id=cid, text=f"Text for {cid}", similarity_score=0.9)


def _rag(*ids: str) -> RAGOutput:
    return RAGOutput(retrieved_clauses=[_clause(i) for i in ids])


def _report(
    decision: str = "approve",
    cited: list[str] | None = None,
    confidence: float = 0.9,
) -> ClaimReport:
    return ClaimReport(
        decision=decision,  # type: ignore[arg-type]
        confidence=confidence,
        cited_clauses=cited if cited is not None else ["COL-001"],
        risk_flags=[],
        reasoning_summary="model proposal",
    )


def _vision_empty_detections() -> VisionOutput:
    return VisionOutput(
        detections=[],
        severity_tier="moderate damage",
        severity_confidence=0.8,
        vqa_answers={"Is the airbag deployed?": "no"},
        low_confidence=False,
    )


def test_valid_citations_accepted():
    out = apply_guardrails(
        _report(cited=["COL-001", "COL-002"]),
        rag=_rag("COL-001", "COL-002", "EXC-001"),
        vision=_vision_empty_detections(),
        verifiers=VerifierOutput(),
        risk=RiskOutput(flags=[], risk_score=0.0),
        extraction_meta={"low_confidence_fields": []},
    )
    assert out.decision == "approve"
    assert out.cited_clauses == ["COL-001", "COL-002"]
    assert 0.0 <= out.confidence <= 1.0


def test_unknown_citation_rejected():
    out = apply_guardrails(
        _report(cited=["COL-001", "FAKE-999"]),
        rag=_rag("COL-001"),
        vision=None,
        verifiers=VerifierOutput(),
        risk=RiskOutput(flags=[], risk_score=0.0),
    )
    assert out.decision == "needs_review"
    assert "FAKE-999" not in out.cited_clauses
    assert "subset" in out.reasoning_summary.lower() or "cited_clauses" in out.reasoning_summary


def test_other_policy_clause_id_rejected_when_not_retrieved():
    # Same ID shape as another policy's clause, but not in this claim's retrieved set.
    out = apply_guardrails(
        _report(cited=["OTH-001"]),
        rag=_rag("COL-001"),
        vision=None,
        verifiers=VerifierOutput(),
        risk=RiskOutput(flags=[], risk_score=0.0),
    )
    assert out.decision == "needs_review"


def test_approve_with_empty_rag_forced_to_review():
    out = apply_guardrails(
        _report(cited=[]),
        rag=RAGOutput(retrieved_clauses=[]),
        vision=None,
        verifiers=VerifierOutput(),
        risk=RiskOutput(flags=[], risk_score=0.0),
    )
    assert out.decision == "needs_review"


def test_approve_with_no_citations_forced_to_review():
    out = apply_guardrails(
        _report(cited=[]),
        rag=_rag("COL-001"),
        vision=None,
        verifiers=VerifierOutput(),
        risk=RiskOutput(flags=[], risk_score=0.0),
    )
    assert out.decision == "needs_review"


def test_deny_with_empty_rag_forced_to_review():
    out = apply_guardrails(
        _report(decision="deny", cited=[]),
        rag=RAGOutput(),
        vision=None,
        verifiers=VerifierOutput(),
        risk=RiskOutput(flags=[], risk_score=0.0),
    )
    assert out.decision == "needs_review"


def test_empty_vision_detections_do_not_force_deny():
    out = apply_guardrails(
        _report(cited=["COL-001"]),
        rag=_rag("COL-001"),
        vision=_vision_empty_detections(),
        verifiers=VerifierOutput(),
        risk=RiskOutput(flags=[], risk_score=0.0),
        extraction_meta={"low_confidence_fields": []},
    )
    assert out.decision == "approve"
    assert out.decision != "deny"


def test_vision_none_does_not_imply_no_damage():
    out = apply_guardrails(
        _report(cited=["COM-001"]),
        rag=_rag("COM-001"),
        vision=None,
        verifiers=VerifierOutput(),
        risk=RiskOutput(flags=[], risk_score=0.0),
        extraction_meta={"low_confidence_fields": []},
    )
    assert out.decision == "approve"


def test_verifier_source_failure_does_not_imply_negative_finding():
    # Approve remains allowed if other evidence is fine; sources_failed only affects confidence.
    base = apply_guardrails(
        _report(cited=["COL-001"]),
        rag=_rag("COL-001"),
        vision=_vision_empty_detections(),
        verifiers=VerifierOutput(sources_failed=[]),
        risk=RiskOutput(flags=[], risk_score=0.0),
        extraction_meta={"low_confidence_fields": []},
    )
    out = apply_guardrails(
        _report(cited=["COL-001"]),
        rag=_rag("COL-001"),
        vision=_vision_empty_detections(),
        verifiers=VerifierOutput(sources_failed=["weather", "nhtsa_vin"]),
        risk=RiskOutput(flags=[], risk_score=0.0),
        extraction_meta={"low_confidence_fields": []},
    )
    assert out.decision == "approve"
    assert out.confidence < base.confidence  # sources_failed penalty applied


def test_high_risk_score_does_not_automatically_deny():
    proposed = _report(decision="needs_review", cited=["COL-001"])
    out = apply_guardrails(
        proposed,
        rag=_rag("COL-001"),
        vision=None,
        verifiers=VerifierOutput(),
        risk=RiskOutput(flags=[], risk_score=0.95),
    )
    assert out.decision == "needs_review"
    assert out.decision != "deny"


def test_material_risk_flag_blocks_approve():
    out = apply_guardrails(
        _report(cited=["COL-001"]),
        rag=_rag("COL-001"),
        vision=None,
        verifiers=VerifierOutput(
            weather_at_incident=WeatherAtIncident(
                condition="Clear", precipitation_mm=0.0, had_storm_event=False
            )
        ),
        risk=RiskOutput(
            flags=[
                RiskFlag(
                    flag_type="weather mismatch",
                    rationale="hail vs clear",
                    severity="medium",
                )
            ],
            risk_score=0.3,
        ),
        extraction_meta={"low_confidence_fields": []},
    )
    assert out.decision == "needs_review"
    assert out.risk_flags[0].flag_type == "weather mismatch"


def test_low_confidence_critical_extraction_blocks_approve():
    out = apply_guardrails(
        _report(cited=["COL-001"]),
        rag=_rag("COL-001"),
        vision=None,
        verifiers=VerifierOutput(),
        risk=RiskOutput(flags=[], risk_score=0.0),
        extraction_meta={"low_confidence_fields": ["policy_id", "coverage_limits.collision"]},
    )
    assert out.decision == "needs_review"


def test_conflicting_evidence_routes_to_review():
    out = apply_guardrails(
        _report(cited=["COM-001"]),
        rag=_rag("COM-001"),
        vision=None,
        verifiers=VerifierOutput(),
        risk=RiskOutput(
            flags=[
                RiskFlag(
                    flag_type="inconsistent claim",
                    rationale="classifier",
                    severity="high",
                )
            ],
            risk_score=0.5,
        ),
    )
    assert out.decision == "needs_review"


def test_risk_flags_copied_from_upstream():
    flags = [
        RiskFlag(flag_type="recall-related damage", rationale="info", severity="info")
    ]
    out = apply_guardrails(
        _report(cited=["COL-001"]),
        rag=_rag("COL-001"),
        vision=None,
        verifiers=VerifierOutput(),
        risk=RiskOutput(flags=flags, risk_score=0.1),
        extraction_meta={"low_confidence_fields": []},
    )
    assert out.decision == "approve"
    assert out.risk_flags == flags


def test_confidence_remains_in_unit_interval():
    score = compute_confidence(
        decision="needs_review",
        rag=RAGOutput(),
        cited_clauses=[],
        vision=VisionOutput(
            detections=[Detection(label="damaged car", confidence=0.9, image_path="x.jpg")],
            severity_tier="severe damage",
            severity_confidence=0.9,
            vqa_answers={},
            low_confidence=False,
        ),
        verifiers=VerifierOutput(sources_failed=["weather", "geocoding", "nhtsa_vin"]),
        risk=RiskOutput(
            flags=[RiskFlag(flag_type="possible staged damage", rationale="x", severity="high")],
            risk_score=1.0,
        ),
        extraction_meta={"low_confidence_fields": ["policy_id", "coverage_limits.collision"]},
        guardrail_override=True,
    )
    assert 0.0 <= score <= 1.0
