"""RAG Agent — retrieve policy clauses hard-filtered by policy_id.

Contract: PROJECT_SPEC.md §6.3 (retrieved_precedents deferred — DECISIONS.md).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.schemas import RAGOutput, RetrievedClause
from api.config import get_settings
from db.models import PolicyClause
from db import session as db_session
from rag.embeddings import embed_query

logger = logging.getLogger(__name__)


def synthesize_query(narrative: str, extracted_fields: dict[str, Any]) -> str:
    """Build retrieval query from narrative + light hints from extracted fields."""
    parts: list[str] = []
    narrative = (narrative or "").strip()
    if narrative:
        parts.append(narrative)

    limits = extracted_fields.get("coverage_limits") or {}
    if isinstance(limits, dict):
        for key in ("collision", "comprehensive", "liability"):
            if key in limits:
                parts.append(f"{key} coverage")

    if extracted_fields.get("deductible") is not None:
        parts.append("deductible")

    line_items = extracted_fields.get("line_items") or []
    if line_items:
        parts.append("repair estimate collision damage")

    if not parts:
        parts.append("auto insurance coverage claim")
    return " ".join(parts)


def run_rag_agent(
    policy_id: str,
    narrative: str,
    extracted_fields: dict[str, Any],
    *,
    db: Session | None = None,
    top_k: int | None = None,
) -> RAGOutput:
    """Retrieve top-k clauses for policy_id by cosine similarity."""
    settings = get_settings()
    k = settings.rag_top_k if top_k is None else top_k

    if not policy_id:
        logger.info("RAG skipped: missing policy_id")
        return RAGOutput(retrieved_clauses=[])

    query_text = synthesize_query(narrative, extracted_fields)
    query_vec = embed_query(query_text)

    owns_session = db is None
    if owns_session:
        db_session.ensure_engine()
        assert db_session.SessionLocal is not None
        db = db_session.SessionLocal()
    assert db is not None

    try:
        distance = PolicyClause.embedding.cosine_distance(query_vec)
        stmt = (
            select(PolicyClause, distance.label("distance"))
            .where(PolicyClause.policy_id == policy_id)
            .order_by(distance)
            .limit(k)
        )
        rows = db.execute(stmt).all()
        retrieved: list[RetrievedClause] = []
        for clause, dist in rows:
            # cosine_distance in [0, 2]; similarity = 1 - distance
            score = float(1.0 - float(dist))
            retrieved.append(
                RetrievedClause(
                    clause_id=clause.clause_id,
                    text=clause.text,
                    similarity_score=score,
                )
            )
        logger.info(
            "RAG policy_id=%s query=%r retrieved=%s",
            policy_id,
            query_text[:120],
            [c.clause_id for c in retrieved],
        )
        return RAGOutput(retrieved_clauses=retrieved)
    finally:
        if owns_session:
            db.close()
