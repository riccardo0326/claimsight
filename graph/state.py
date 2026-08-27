"""Typed ClaimState for the LangGraph orchestrator (Slice 9)."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from agents.schemas import (
    ClaimReport,
    DocumentOutput,
    RAGOutput,
    RiskOutput,
    VerifierOutput,
    VisionOutput,
)


class ClaimState(TypedDict):
    """Shared graph state. Input fields are required; agent outputs accumulate.

    The SQLAlchemy session is *not* stored here — pass it via invoke config
    (see DECISIONS.md D41).
    """

    claim_id: str
    narrative: str
    incident_location: str | None
    policy_pdf: str
    estimate_pdf: str
    image_paths: list[str]
    document: NotRequired[DocumentOutput]
    extraction_meta: NotRequired[dict[str, Any]]
    vision: NotRequired[VisionOutput | None]
    verifiers: NotRequired[VerifierOutput]
    rag: NotRequired[RAGOutput]
    risk: NotRequired[RiskOutput]
    adjudication: NotRequired[ClaimReport]
