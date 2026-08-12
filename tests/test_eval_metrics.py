"""Unit tests for Slice 6 eval metrics."""

from __future__ import annotations

from agents.schemas import ClaimReport, RAGOutput, RetrievedClause, RiskFlag, RiskOutput
from eval.metrics import (
    CaseScore,
    aggregate_metrics,
    is_hallucinated,
    predicted_fraud_flag,
    score_case,
)
from eval.schema import GoldenCase, GroundTruth, UpstreamSnapshot
from agents.schemas import DocumentOutput


def _rag(*ids: str) -> RAGOutput:
    return RAGOutput(
        retrieved_clauses=[
            RetrievedClause(clause_id=i, text=f"text {i}", similarity_score=0.9)
            for i in ids
        ]
    )


def _case(
    *,
    decision: str = "approve",
    clause_ids: list[str] | None = None,
    fraud_flag: bool = False,
    rag: RAGOutput | None = None,
    risk: RiskOutput | None = None,
) -> GoldenCase:
    return GoldenCase(
        claim_id="t1",
        narrative="test",
        upstream=UpstreamSnapshot(
            document_agent=DocumentOutput(policy_id="POL-1"),
            rag=rag or _rag("COL-001"),
            risk=risk or RiskOutput(risk_score=0.0),
        ),
        ground_truth=GroundTruth(
            decision=decision,  # type: ignore[arg-type]
            clause_ids=clause_ids or ["COL-001"],
            fraud_flag=fraud_flag,
        ),
    )


def test_predicted_fraud_flag_material_only():
    report = ClaimReport(
        decision="needs_review",
        confidence=0.4,
        cited_clauses=[],
        risk_flags=[
            RiskFlag(flag_type="weather mismatch", rationale="x", severity="medium")
        ],
        reasoning_summary="r",
    )
    assert predicted_fraud_flag(report) is True

    clean = ClaimReport(
        decision="approve",
        confidence=0.7,
        cited_clauses=["COL-001"],
        risk_flags=[],
        reasoning_summary="r",
    )
    assert predicted_fraud_flag(clean) is False


def test_is_hallucinated_detects_unknown_cite():
    rag = _rag("COL-001")
    bad = ClaimReport(
        decision="needs_review",
        confidence=0.3,
        cited_clauses=["OTHER-COL-001"],
        risk_flags=[],
        reasoning_summary="r",
    )
    assert is_hallucinated(bad, rag) is True

    good = ClaimReport(
        decision="approve",
        confidence=0.7,
        cited_clauses=["COL-001"],
        risk_flags=[],
        reasoning_summary="r",
    )
    assert is_hallucinated(good, rag) is False


def test_score_case_decision_match():
    case = _case(decision="approve")
    report = ClaimReport(
        decision="approve",
        confidence=0.8,
        cited_clauses=["COL-001"],
        risk_flags=[],
        reasoning_summary="ok",
    )
    scored = score_case(case, report)
    assert scored.decision_match is True
    assert scored.hallucinated is False


def test_aggregate_metrics_perfect_and_regress():
    perfect = [
        CaseScore("a", True, False, False, False),
        CaseScore("b", True, False, True, True),
    ]
    m = aggregate_metrics(perfect)
    assert m.n_cases == 2
    assert m.decision_accuracy == 1.0
    assert m.hallucination_rate == 0.0
    assert m.fraud_precision == 1.0
    assert m.fraud_recall == 1.0

    regress = [
        CaseScore("a", False, True, True, False),  # FP + halluc + miss
        CaseScore("b", True, False, False, True),  # FN
    ]
    m2 = aggregate_metrics(regress)
    assert m2.decision_accuracy == 0.5
    assert m2.hallucination_rate == 0.5
    assert m2.fraud_precision == 0.0
    assert m2.fraud_recall == 0.0


def test_aggregate_metrics_empty():
    m = aggregate_metrics([])
    assert m.n_cases == 0
    assert m.fraud_precision is None
    assert m.fraud_recall is None
