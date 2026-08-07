from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    cost: float


class DocumentOutput(BaseModel):
    """Document Agent contract from PROJECT_SPEC.md §6.2.

    policy_id and deductible are Optional so low-confidence / failed
    extractions can return None rather than hallucinated values.
    """

    policy_id: str | None = None
    coverage_limits: dict[str, float] = Field(default_factory=dict)
    deductible: float | None = None
    vin: str | None = None
    incident_date: date | None = None
    line_items: list[LineItem] = Field(default_factory=list)


class RetrievedClause(BaseModel):
    clause_id: str
    text: str
    similarity_score: float


class RAGOutput(BaseModel):
    """RAG Agent contract from PROJECT_SPEC.md §6.3.

    retrieved_precedents is deliberately omitted — deferred to a later slice
    (see DECISIONS.md). Do not stub with fake data.
    """

    retrieved_clauses: list[RetrievedClause] = Field(default_factory=list)


class Detection(BaseModel):
    label: str
    confidence: float
    image_path: str


class VisionOutput(BaseModel):
    """Vision Agent contract (Slice 3). See DECISIONS.md D14 for deviations
    from PROJECT_SPEC.md §6.1 (zero-shot models; no bbox; image_path on detections).
    """

    detections: list[Detection] = Field(default_factory=list)
    severity_tier: str
    severity_confidence: float
    vqa_answers: dict[str, str] = Field(default_factory=dict)
    low_confidence: bool
