"""Pydantic models for the Slice 6 golden dataset and eval report."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.schemas import (
    DocumentOutput,
    RAGOutput,
    RiskOutput,
    VerifierOutput,
    VisionOutput,
)


class GroundTruth(BaseModel):
    """Labels for a golden claim (extends PROJECT_SPEC §8.1)."""

    decision: Literal["approve", "deny", "needs_review"]
    clause_ids: list[str] = Field(default_factory=list)
    fraud_flag: bool = False


class UpstreamSnapshot(BaseModel):
    """Canned upstream agent outputs for Adjudicator-only eval."""

    document_agent: DocumentOutput
    extraction_meta: dict[str, Any] = Field(default_factory=dict)
    vision: VisionOutput | None = None
    verifiers: VerifierOutput = Field(default_factory=VerifierOutput)
    rag: RAGOutput = Field(default_factory=RAGOutput)
    risk: RiskOutput = Field(default_factory=lambda: RiskOutput(risk_score=0.0))


class GoldenCase(BaseModel):
    """One row of fixtures/golden/manifest.jsonl."""

    claim_id: str
    narrative: str
    policy_pdf: str = "fixtures/sample_policy.pdf"
    estimate_pdf: str = "fixtures/sample_estimate.pdf"
    images: list[str] = Field(default_factory=list)
    incident_location: str | None = None
    upstream: UpstreamSnapshot
    ground_truth: GroundTruth
    notes: str = ""


class CaseResult(BaseModel):
    claim_id: str
    predicted_decision: str
    ground_truth_decision: str
    decision_match: bool
    hallucinated: bool
    predicted_fraud_flag: bool
    ground_truth_fraud_flag: bool
    cited_clauses: list[str] = Field(default_factory=list)
    confidence: float | None = None
    reasoning_summary: str = ""


class EvalMetrics(BaseModel):
    n_cases: int
    decision_accuracy: float
    hallucination_rate: float
    fraud_precision: float | None
    fraud_recall: float | None
    fraud_true_positives: int = 0
    fraud_false_positives: int = 0
    fraud_false_negatives: int = 0
    fraud_true_negatives: int = 0


class EvalReport(BaseModel):
    mode: str
    prompt_version: str = "prompts/adjudicator_v1.md"
    model: str | None = None
    metrics: EvalMetrics
    cases: list[CaseResult] = Field(default_factory=list)
