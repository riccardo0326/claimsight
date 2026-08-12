"""Adjudicator Agent — frontier LLM synthesis + deterministic guardrails (Slice 5).

Contract: ClaimReport in agents/schemas.py. See PROJECT_SPEC.md §6.6 / DECISIONS.md.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agents.adjudicator_guardrails import apply_guardrails, fallback_needs_review
from agents.llm_openai import LLMError, complete_json
from agents.schemas import (
    ClaimReport,
    DocumentOutput,
    RAGOutput,
    RiskOutput,
    VerifierOutput,
    VisionOutput,
)

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "adjudicator_v1.md"

SYSTEM_PROMPT = """You are the ClaimSight Adjudicator. You synthesize upstream claim evidence into a structured first-pass recommendation for a human adjuster.

Return ONLY a single JSON object with exactly these fields:
{
  "decision": "approve" | "deny" | "needs_review",
  "confidence": 0.0,
  "cited_clauses": ["CLAUSE_ID"],
  "risk_flags": [],
  "reasoning_summary": "..."
}

Rules:
1. Policy evidence: Only retrieved RAG clauses in the user message may be used as policy evidence. Do NOT invent policy clauses, exclusions, coverage terms, deductibles, policy language, or clause IDs.
2. Citations: cited_clauses MUST be a subset of the provided rag.retrieved_clauses[].clause_id values. Never create a citation ID.
3. approve only when retrieved policy evidence supports coverage and there is no unresolved critical contradiction. Every approve MUST cite at least one clause.
4. deny only when retrieved policy evidence supports an exclusion/denial. Do NOT deny merely because evidence is missing, Vision detections are empty, a verifier failed, risk_score is high, or the narrative is incomplete.
5. needs_review when evidence is insufficient, signals conflict, critical extraction is unreliable, verifier info is missing for a material check, or risk signals create unresolved uncertainty.
6. Vision: detections=[] means NO DETECTION SIGNAL, not "no damage". vision=null means no photos / no Vision analysis. Severity/VQA may still provide evidence when detections are empty. Never infer absence of damage from empty detections or null vision alone.
7. Verifiers: sources_failed means MISSING EVIDENCE, not negative evidence (e.g. weather failure does not mean no storm).
8. Fraud/Risk: risk_flags and risk_score are signals, NOT proof of fraud. High risk must not automatically become deny; bias ambiguous cases toward needs_review.
9. In reasoning_summary, distinguish FACT/EVIDENCE vs INFERENCE vs UNKNOWN/MISSING. Do not present an inference as quoted policy evidence.
10. Set confidence to any float in [0, 1]; the system will replace it with a deterministic heuristic. Prefer leaving risk_flags as [] — the system overwrites them from upstream Fraud/Risk output.
"""


def _load_system_prompt() -> str:
    """Prefer the versioned prompt file; fall back to embedded SYSTEM_PROMPT."""
    try:
        text = PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return SYSTEM_PROMPT
    # Extract the fenced block after "## System" if present; else use embedded.
    match = re.search(
        r"## System\s+(.*?)(?=\n## User template|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return SYSTEM_PROMPT
    body = match.group(1).strip()
    # Strip leading markdown fluff; keep the instructional prose.
    # The .md file mixes prose + a json example; use embedded canonical rules
    # for the actual API call to avoid sending markdown noise.
    return SYSTEM_PROMPT


def build_user_prompt(
    *,
    narrative: str,
    document: DocumentOutput,
    extraction_meta: dict[str, Any],
    vision: VisionOutput | None,
    rag: RAGOutput,
    verifiers: VerifierOutput,
    risk: RiskOutput,
) -> str:
    legal_ids = [c.clause_id for c in rag.retrieved_clauses]
    payload = {
        "narrative": narrative or "",
        "document_agent": document.model_dump(mode="json"),
        "extraction_meta": extraction_meta,
        "vision": vision.model_dump(mode="json") if vision is not None else None,
        "rag": rag.model_dump(mode="json"),
        "verifiers": verifiers.model_dump(mode="json"),
        "risk": risk.model_dump(mode="json"),
        "legal_citation_ids": legal_ids,
        "vision_semantics": {
            "detections_empty_means": "NO_DETECTION_SIGNAL_not_no_damage",
            "vision_null_means": "no_photos_no_vision_analysis",
        },
        "verifier_semantics": {
            "sources_failed_means": "MISSING_EVIDENCE_not_negative_evidence",
        },
        "risk_semantics": {
            "risk_score_means": "heuristic_signal_not_proof_of_fraud",
        },
    }
    return (
        "Adjudicate this claim. Use ONLY the evidence below.\n"
        "Return ONLY the ClaimReport JSON object.\n\n"
        f"{json.dumps(payload, indent=2, default=str)}\n"
    )


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        # Strip optional markdown fences.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def parse_claim_report(raw: str) -> ClaimReport:
    """Parse LLM text into ClaimReport. Raises ValueError/ValidationError on failure."""
    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        raise ValueError("LLM JSON root must be an object")
    return ClaimReport.model_validate(data)


def run_adjudicator(
    *,
    narrative: str,
    document: DocumentOutput,
    extraction_meta: dict[str, Any],
    vision: VisionOutput | None,
    rag: RAGOutput,
    verifiers: VerifierOutput,
    risk: RiskOutput,
    llm_complete: Callable[[list[dict[str, str]]], str] | None = None,
) -> ClaimReport:
    """Propose via LLM (or inject llm_complete), then apply deterministic guardrails."""
    messages = [
        {"role": "system", "content": _load_system_prompt()},
        {
            "role": "user",
            "content": build_user_prompt(
                narrative=narrative,
                document=document,
                extraction_meta=extraction_meta,
                vision=vision,
                rag=rag,
                verifiers=verifiers,
                risk=risk,
            ),
        },
    ]

    complete = llm_complete or (lambda msgs: complete_json(msgs))

    try:
        raw = complete(messages)
    except LLMError as exc:
        logger.warning("Adjudicator LLM failure: %s", exc)
        return fallback_needs_review(
            reason=f"LLM failure: {exc}",
            risk=risk,
            rag=rag,
            vision=vision,
            verifiers=verifiers,
            extraction_meta=extraction_meta,
        )
    except Exception as exc:  # noqa: BLE001 — never fail the claim on LLM errors
        logger.exception("Adjudicator unexpected LLM error")
        return fallback_needs_review(
            reason=f"LLM unexpected error: {exc}",
            risk=risk,
            rag=rag,
            vision=vision,
            verifiers=verifiers,
            extraction_meta=extraction_meta,
        )

    try:
        proposed = parse_claim_report(raw)
    except (json.JSONDecodeError, ValueError, ValidationError, TypeError) as exc:
        logger.info("Adjudicator schema/parse failure: %s", exc)
        return fallback_needs_review(
            reason=f"schema/parse validation failed: {exc}",
            risk=risk,
            rag=rag,
            vision=vision,
            verifiers=verifiers,
            extraction_meta=extraction_meta,
        )

    return apply_guardrails(
        proposed,
        rag=rag,
        vision=vision,
        verifiers=verifiers,
        risk=risk,
        extraction_meta=extraction_meta,
    )
