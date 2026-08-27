"""Graph node wrappers around existing agent functions + Langfuse spans."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agents.adjudicator import run_adjudicator
from agents.document_agent import run_document_agent
from agents.fraud_agent import run_fraud_agent
from agents.observability import span, update_span
from agents.rag_agent import run_rag_agent
from agents.verifiers import run_verifiers
from agents.vision_agent import run_vision_agent
from graph.state import ClaimState


def _db(config: RunnableConfig | None) -> Any:
    if not config:
        return None
    return (config.get("configurable") or {}).get("db")


def document_node(state: ClaimState) -> dict[str, Any]:
    policy_path = state["policy_pdf"]
    estimate_path = state["estimate_pdf"]
    with span(
        "document",
        input={"policy_pdf": policy_path, "estimate_pdf": estimate_path},
    ) as doc_span:
        output, meta = run_document_agent(policy_path, estimate_path)
        update_span(
            doc_span,
            output={
                "policy_id": output.policy_id,
                "vin": output.vin,
                "line_item_count": len(output.line_items),
                "low_confidence_fields": meta.get("low_confidence_fields"),
            },
        )
    return {"document": output, "extraction_meta": meta}


def vision_node(state: ClaimState) -> dict[str, Any]:
    image_paths = state.get("image_paths") or []
    with span(
        "vision",
        input={"photo_count": len(image_paths)},
        metadata={"skipped": not bool(image_paths)},
    ) as vision_span:
        vision_out = run_vision_agent(image_paths)
        update_span(
            vision_span,
            output=(
                {
                    "detection_count": len(vision_out.detections),
                    "severity_tier": vision_out.severity_tier,
                    "low_confidence": vision_out.low_confidence,
                }
                if vision_out is not None
                else {"skipped": True}
            ),
        )
    return {"vision": vision_out}


def verifiers_node(state: ClaimState, config: RunnableConfig) -> dict[str, Any]:
    document = state["document"]
    location = state.get("incident_location")
    with span("verifiers", input={"incident_location": location}) as ver_span:
        verifier_out = run_verifiers(
            document,
            incident_location=location,
            db=_db(config),
        )
        update_span(
            ver_span,
            output={
                "make": verifier_out.make,
                "model": verifier_out.model,
                "model_year": verifier_out.model_year,
                "recall_count": len(verifier_out.nhtsa_recalls),
                "sources_failed": verifier_out.sources_failed,
            },
        )
    return {"verifiers": verifier_out}


def rag_node(state: ClaimState, config: RunnableConfig) -> dict[str, Any]:
    document = state["document"]
    narrative = state.get("narrative") or ""
    doc_dump = document.model_dump(mode="json")
    with span(
        "rag",
        input={
            "policy_id": document.policy_id or "",
            "narrative_len": len(narrative),
        },
    ) as rag_span:
        rag_out = run_rag_agent(
            policy_id=document.policy_id or "",
            narrative=narrative,
            extracted_fields=doc_dump,
            db=_db(config),
        )
        update_span(
            rag_span,
            output={
                "retrieved_count": len(rag_out.retrieved_clauses),
                "clause_ids": [c.clause_id for c in rag_out.retrieved_clauses],
            },
        )
    return {"rag": rag_out}


def fraud_risk_node(state: ClaimState) -> dict[str, Any]:
    narrative = state.get("narrative") or ""
    with span("fraud_risk", input={"narrative_len": len(narrative)}) as risk_span:
        risk_out = run_fraud_agent(
            narrative,
            state["document"],
            state["verifiers"],
        )
        update_span(
            risk_span,
            output={
                "risk_score": risk_out.risk_score,
                "flag_types": [f.flag_type for f in risk_out.flags],
            },
        )
    return {"risk": risk_out}


def adjudicator_node(state: ClaimState) -> dict[str, Any]:
    missing = [
        key
        for key in ("document", "extraction_meta", "rag", "verifiers", "risk")
        if key not in state
    ]
    if "vision" not in state:
        missing.append("vision")
    if missing:
        raise RuntimeError(f"adjudicator missing ClaimState keys: {missing}")

    with span("adjudicator") as adj_span:
        adjudication_out = run_adjudicator(
            narrative=state.get("narrative") or "",
            document=state["document"],
            extraction_meta=state["extraction_meta"],
            vision=state.get("vision"),
            rag=state["rag"],
            verifiers=state["verifiers"],
            risk=state["risk"],
        )
        update_span(
            adj_span,
            output={
                "decision": adjudication_out.decision,
                "confidence": adjudication_out.confidence,
                "cited_clauses": adjudication_out.cited_clauses,
            },
        )
    return {"adjudication": adjudication_out}
