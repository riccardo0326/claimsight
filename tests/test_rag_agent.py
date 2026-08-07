"""Unit tests for the RAG Agent (collision weighting + policy scope)."""

from __future__ import annotations

from pathlib import Path

from agents.rag_agent import run_rag_agent
from rag.ingest import ingest_file

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_collision_narrative_retrieves_collision_clauses(pg_session, sample_clauses):
    ingest_file(FIXTURES / "sample_policy_clauses.json", db=pg_session)
    policy_id = sample_clauses["policy_id"]
    sample_ids = {c["clause_id"] for c in sample_clauses["clauses"]}
    type_by_id = {c["clause_id"]: c["clause_type"] for c in sample_clauses["clauses"]}

    result = run_rag_agent(
        policy_id=policy_id,
        narrative=(
            "The insured vehicle was damaged in a front-end collision with another car. "
            "Please confirm collision coverage and repair payment after deductible."
        ),
        extracted_fields={
            "policy_id": policy_id,
            "coverage_limits": {"collision": 50000.0},
            "line_items": [{"description": "Front bumper replacement", "cost": 850.0}],
        },
        db=pg_session,
        top_k=5,
    )

    assert result.retrieved_clauses
    returned_ids = [c.clause_id for c in result.retrieved_clauses]
    assert all(cid in sample_ids for cid in returned_ids)

    # Top results should skew toward collision-type clauses.
    top_types = [type_by_id[cid] for cid in returned_ids[:3]]
    assert top_types.count("collision") >= 1
    assert "collision" in top_types

    # Similarity scores should be finite and ordered descending.
    scores = [c.similarity_score for c in result.retrieved_clauses]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_missing_policy_id_returns_empty(pg_session):
    result = run_rag_agent(
        policy_id="",
        narrative="collision claim",
        extracted_fields={},
        db=pg_session,
    )
    assert result.retrieved_clauses == []
