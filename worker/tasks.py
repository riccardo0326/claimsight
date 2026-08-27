"""Celery tasks for claim processing."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from agents.observability import trace_claim
from db.models import Claim, ClaimStatus
from db import session as db_session
from graph.claim_graph import invoke_claim_graph, result_from_state
from graph.state import ClaimState
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="worker.tasks.process_claim", bind=True, max_retries=0)
def process_claim(self, claim_id: str) -> dict:
    """Load claim, invoke LangGraph, persist result (status=completed on success)."""
    db_session.ensure_engine()
    assert db_session.SessionLocal is not None
    db = db_session.SessionLocal()
    try:
        try:
            claim_uuid = uuid.UUID(claim_id)
        except ValueError:
            logger.error("Invalid claim_id=%s", claim_id)
            return {"status": "failed", "error": "invalid claim_id"}

        claim = db.get(Claim, claim_uuid)
        if claim is None:
            logger.error("Claim not found: %s", claim_id)
            return {"status": "failed", "error": "claim not found"}

        claim.status = ClaimStatus.processing
        claim.updated_at = datetime.now(timezone.utc)
        db.commit()

        image_paths = claim.input_paths.get("damage_photos") or []
        initial: ClaimState = {
            "claim_id": claim_id,
            "narrative": claim.narrative or "",
            "incident_location": claim.incident_location,
            "policy_pdf": claim.input_paths["policy_pdf"],
            "estimate_pdf": claim.input_paths["estimate_pdf"],
            "image_paths": image_paths,
        }

        with trace_claim(claim_id):
            final = invoke_claim_graph(initial, db=db)
            claim.result = result_from_state(final)
            claim.status = ClaimStatus.completed
            claim.updated_at = datetime.now(timezone.utc)
            db.commit()
            decision = final["adjudication"].decision
            logger.info("Claim %s completed decision=%s", claim_id, decision)
            return {"status": "completed", "claim_id": claim_id}
    except Exception as exc:  # noqa: BLE001 — persist failure then re-raise for Celery logs
        logger.exception("Claim %s failed: %s", claim_id, exc)
        db.rollback()
        try:
            claim = db.get(Claim, uuid.UUID(claim_id)) if claim_id else None
            if claim is not None:
                claim.status = ClaimStatus.failed
                claim.result = {"error": str(exc)}
                claim.updated_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to persist failure state for claim %s", claim_id)
        return {"status": "failed", "claim_id": claim_id, "error": str(exc)}
    finally:
        db.close()
