"""Unit tests for policy clause ingestion into pgvector."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from db.models import PolicyClause
from rag.ingest import ingest_file

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_ingest_sample_clauses_writes_embeddings(pg_session, sample_clauses):
    count = ingest_file(FIXTURES / "sample_policy_clauses.json", db=pg_session)
    assert count == len(sample_clauses["clauses"])

    rows = pg_session.scalars(select(PolicyClause)).all()
    assert len(rows) == count
    for row in rows:
        assert row.policy_id == sample_clauses["policy_id"]
        assert row.embedding is not None
        assert len(row.embedding) == 384
        assert row.clause_id
        assert row.text
