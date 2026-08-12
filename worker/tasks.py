"""Celery tasks for claim processing."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from agents.adjudicator import run_adjudicator
from agents.document_agent import run_document_agent
from agents.fraud_agent import run_fraud_agent
from agents.rag_agent import run_rag_agent
from agents.verifiers import run_verifiers
from agents.vision_agent import run_vision_agent
from db.models import Claim, ClaimStatus
from db import session as db_session
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="worker.tasks.process_claim", bind=True, max_retries=0)
def process_claim(self, claim_id: str) -> dict:
    """Load claim, run Document → Vision → Verifiers → RAG → Fraud/Risk → Adjudicator, persist."""
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

        policy_path = claim.input_paths["policy_pdf"]
        estimate_path = claim.input_paths["estimate_pdf"]

        output, meta = run_document_agent(policy_path, estimate_path)
        doc_dump = output.model_dump(mode="json")

        image_paths = claim.input_paths.get("damage_photos") or []
        vision_out = run_vision_agent(image_paths) if image_paths else None

        verifier_out = run_verifiers(
            output,
            incident_location=claim.incident_location,
            db=db,
        )

        rag_out = run_rag_agent(
            policy_id=output.policy_id or "",
            narrative=claim.narrative or "",
            extracted_fields=doc_dump,
            db=db,
        )

        risk_out = run_fraud_agent(
            claim.narrative or "",
            output,
            verifier_out,
        )

        adjudication_out = run_adjudicator(
            narrative=claim.narrative or "",
            document=output,
            extraction_meta=meta,
            vision=vision_out,
            rag=rag_out,
            verifiers=verifier_out,
            risk=risk_out,
        )

        claim.result = {
            "document_agent": doc_dump,
            "extraction_meta": meta,
            "vision": vision_out.model_dump(mode="json") if vision_out else None,
            "verifiers": verifier_out.model_dump(mode="json"),
            "rag": rag_out.model_dump(mode="json"),
            "risk": risk_out.model_dump(mode="json"),
            "adjudication": adjudication_out.model_dump(mode="json"),
        }
        claim.status = ClaimStatus.completed
        claim.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(
            "Claim %s completed decision=%s",
            claim_id,
            adjudication_out.decision,
        )
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
