"""Cross-policy leak test — retrieval must never return another policy's clauses."""

from __future__ import annotations

from pathlib import Path

from agents.rag_agent import run_rag_agent
from rag.ingest import ingest_file

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_retrieval_never_leaks_across_policies(pg_session, sample_clauses, other_clauses):
    ingest_file(FIXTURES / "sample_policy_clauses.json", db=pg_session)
    ingest_file(FIXTURES / "other_policy_clauses.json", db=pg_session)

    other_ids = {c["clause_id"] for c in other_clauses["clauses"]}
    sample_ids = {c["clause_id"] for c in sample_clauses["clauses"]}

    # Narrative deliberately uses collision wording that also appears in OTHER policy.
    result = run_rag_agent(
        policy_id=sample_clauses["policy_id"],
        narrative=(
            "Front-end collision damage to bumper and body panels; "
            "need collision coverage under the policy."
        ),
        extracted_fields={
            "policy_id": sample_clauses["policy_id"],
            "coverage_limits": {"collision": 50000.0},
        },
        db=pg_session,
        top_k=5,
    )

    assert result.retrieved_clauses, "expected non-empty retrieval for sample policy"
    returned = {c.clause_id for c in result.retrieved_clauses}
    assert returned.isdisjoint(other_ids), f"leaked other-policy clauses: {returned & other_ids}"
    assert returned.issubset(sample_ids)
