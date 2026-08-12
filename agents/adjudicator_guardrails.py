"""Deterministic Adjudicator guardrails + confidence heuristic (Slice 5).

The LLM proposes a ClaimReport; these rules decide what is admissible.
See DECISIONS.md Slice 5.
"""

from __future__ import annotations

from typing import Any

from agents.schemas import (
    ClaimReport,
    RAGOutput,
    RiskFlag,
    RiskOutput,
    VerifierOutput,
    VisionOutput,
)

# Risk flag types that block an automatic approve (signal → review, not deny).
MATERIAL_RISK_FLAG_TYPES = frozenset(
    {
        "weather mismatch",
        "possible staged damage",
        "inconsistent claim",
    }
)


def retrieved_clause_ids(rag: RAGOutput) -> set[str]:
    return {c.clause_id for c in rag.retrieved_clauses}


def citations_valid(cited: list[str], rag: RAGOutput) -> bool:
    return set(cited) <= retrieved_clause_ids(rag)


def critical_low_confidence_fields(extraction_meta: dict[str, Any] | None) -> list[str]:
    meta = extraction_meta or {}
    low = meta.get("low_confidence_fields") or []
    if not isinstance(low, list):
        return []
    out: list[str] = []
    for field in low:
        if not isinstance(field, str):
            continue
        if field == "policy_id" or field.startswith("coverage_limits."):
            out.append(field)
    return out


def has_material_risk_flags(flags: list[RiskFlag]) -> bool:
    return any(f.flag_type in MATERIAL_RISK_FLAG_TYPES for f in flags)


def compute_confidence(
    *,
    decision: str,
    rag: RAGOutput,
    cited_clauses: list[str],
    vision: VisionOutput | None,
    verifiers: VerifierOutput,
    risk: RiskOutput,
    extraction_meta: dict[str, Any] | None,
    guardrail_override: bool,
) -> float:
    """Non-calibrated evidence-quality heuristic in [0, 1]. See DECISIONS.md."""
    score = 0.55
    cites_ok = citations_valid(cited_clauses, rag)

    if rag.retrieved_clauses and cites_ok and cited_clauses:
        score += 0.15
    elif rag.retrieved_clauses and cites_ok and decision != "approve":
        # Deny/review may cite none; still credit usable RAG context lightly.
        score += 0.05

    if vision is not None and (vision.detections or not vision.low_confidence):
        score += 0.10

    if critical_low_confidence_fields(extraction_meta):
        score -= 0.15

    if verifiers.sources_failed:
        score -= 0.10

    if guardrail_override or decision == "needs_review":
        score -= 0.20

    if any(f.severity in {"medium", "high"} for f in risk.flags):
        score -= 0.10

    return max(0.0, min(1.0, round(score, 4)))


def _force_review(
    proposed: ClaimReport,
    *,
    reason: str,
    risk: RiskOutput,
    rag: RAGOutput,
    vision: VisionOutput | None,
    verifiers: VerifierOutput,
    extraction_meta: dict[str, Any] | None,
) -> ClaimReport:
    # Keep only citations that are still in the retrieved set (drop inventions).
    legal = [c for c in proposed.cited_clauses if c in retrieved_clause_ids(rag)]
    confidence = compute_confidence(
        decision="needs_review",
        rag=rag,
        cited_clauses=legal,
        vision=vision,
        verifiers=verifiers,
        risk=risk,
        extraction_meta=extraction_meta,
        guardrail_override=True,
    )
    summary = (
        f"Guardrail rejected model output: {reason}. "
        f"Original model decision was {proposed.decision!r}. "
        f"Model summary (for audit): {proposed.reasoning_summary}"
    )
    return ClaimReport(
        decision="needs_review",
        confidence=confidence,
        cited_clauses=legal,
        risk_flags=list(risk.flags),
        reasoning_summary=summary,
    )


def apply_guardrails(
    proposed: ClaimReport,
    *,
    rag: RAGOutput,
    vision: VisionOutput | None,
    verifiers: VerifierOutput,
    risk: RiskOutput,
    extraction_meta: dict[str, Any] | None = None,
) -> ClaimReport:
    """Validate / possibly override an LLM-proposed ClaimReport.

    Always overwrites risk_flags from upstream RiskOutput.
    Never auto-denies solely for empty vision detections, verifier failures,
    or high risk_score.
    """
    retrieved = retrieved_clause_ids(rag)
    cited = list(proposed.cited_clauses)

    # Guardrail A — citation subset
    if not set(cited) <= retrieved:
        return _force_review(
            proposed,
            reason="cited_clauses is not a subset of retrieved clause_ids",
            risk=risk,
            rag=rag,
            vision=vision,
            verifiers=verifiers,
            extraction_meta=extraction_meta,
        )

    # Guardrail B/C — approve requires usable policy evidence + ≥1 citation
    if proposed.decision == "approve":
        if not rag.retrieved_clauses:
            return _force_review(
                proposed,
                reason="approve rejected: RAG retrieved_clauses is empty",
                risk=risk,
                rag=rag,
                vision=vision,
                verifiers=verifiers,
                extraction_meta=extraction_meta,
            )
        if not cited:
            return _force_review(
                proposed,
                reason="approve rejected: no policy citations provided",
                risk=risk,
                rag=rag,
                vision=vision,
                verifiers=verifiers,
                extraction_meta=extraction_meta,
            )

    # Guardrail F/H — material risk / weather mismatch blocks approve
    if proposed.decision == "approve" and has_material_risk_flags(risk.flags):
        return _force_review(
            proposed,
            reason=(
                "approve rejected: material risk flags present "
                f"({sorted({f.flag_type for f in risk.flags if f.flag_type in MATERIAL_RISK_FLAG_TYPES})}); "
                "risk is a signal toward needs_review, not proof of fraud"
            ),
            risk=risk,
            rag=rag,
            vision=vision,
            verifiers=verifiers,
            extraction_meta=extraction_meta,
        )

    # Guardrail G — low-confidence critical extraction blocks approve
    if proposed.decision == "approve":
        critical = critical_low_confidence_fields(extraction_meta)
        if critical:
            return _force_review(
                proposed,
                reason=(
                    "approve rejected: low-confidence critical extraction fields "
                    f"{critical}"
                ),
                risk=risk,
                rag=rag,
                vision=vision,
                verifiers=verifiers,
                extraction_meta=extraction_meta,
            )

    # Guardrail H — empty RAG with non-review decision that asserts policy certainty
    # (deny without RAG is also uncertain unless we already have review)
    if not rag.retrieved_clauses and proposed.decision == "deny":
        return _force_review(
            proposed,
            reason="deny rejected: no retrieved policy evidence to support denial",
            risk=risk,
            rag=rag,
            vision=vision,
            verifiers=verifiers,
            extraction_meta=extraction_meta,
        )

    # Accepted path — recompute confidence; always copy upstream risk flags.
    # Note: empty detections / vision=None / sources_failed do NOT force deny.
    confidence = compute_confidence(
        decision=proposed.decision,
        rag=rag,
        cited_clauses=cited,
        vision=vision,
        verifiers=verifiers,
        risk=risk,
        extraction_meta=extraction_meta,
        guardrail_override=False,
    )
    return ClaimReport(
        decision=proposed.decision,
        confidence=confidence,
        cited_clauses=cited,
        risk_flags=list(risk.flags),
        reasoning_summary=proposed.reasoning_summary,
    )


def fallback_needs_review(
    *,
    reason: str,
    risk: RiskOutput,
    rag: RAGOutput | None = None,
    vision: VisionOutput | None = None,
    verifiers: VerifierOutput | None = None,
    extraction_meta: dict[str, Any] | None = None,
) -> ClaimReport:
    """Deterministic report when LLM/schema validation fails."""
    rag = rag or RAGOutput()
    verifiers = verifiers or VerifierOutput()
    confidence = compute_confidence(
        decision="needs_review",
        rag=rag,
        cited_clauses=[],
        vision=vision,
        verifiers=verifiers,
        risk=risk,
        extraction_meta=extraction_meta,
        guardrail_override=True,
    )
    return ClaimReport(
        decision="needs_review",
        confidence=confidence,
        cited_clauses=[],
        risk_flags=list(risk.flags),
        reasoning_summary=f"Adjudicator output rejected: {reason}",
    )
