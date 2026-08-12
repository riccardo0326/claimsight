"""Integration tests for Slice 5 pipeline (mocked upstream + mocked LLM)."""

from __future__ import annotations

import json

from agents.schemas import (
    ClaimReport,
    RAGOutput,
    RetrievedClause,
    RiskOutput,
    VerifierOutput,
    VisionOutput,
)


def test_pipeline_persists_adjudication_key(
    client, policy_pdf, estimate_pdf, monkeypatch
):
    """Full worker path with mocked agents + mocked adjudicator."""

    def fake_adjudicator(**_kwargs):
        return ClaimReport(
            decision="approve",
            confidence=0.7,
            cited_clauses=["COL-001"],
            risk_flags=[],
            reasoning_summary="Integration stub approve.",
        )

    monkeypatch.setattr(
        "worker.tasks.run_verifiers",
        lambda *_a, **_k: VerifierOutput(make="HONDA", model="Accord", model_year=2003),
    )
    monkeypatch.setattr(
        "worker.tasks.run_fraud_agent",
        lambda *_a, **_k: RiskOutput(flags=[], risk_score=0.05),
    )
    monkeypatch.setattr(
        "worker.tasks.run_rag_agent",
        lambda **_k: RAGOutput(
            retrieved_clauses=[
                RetrievedClause(
                    clause_id="COL-001",
                    text="Collision coverage applies.",
                    similarity_score=0.9,
                )
            ]
        ),
    )
    monkeypatch.setattr("worker.tasks.run_adjudicator", fake_adjudicator)

    files = {
        "policy_pdf": ("sample_policy.pdf", policy_pdf.read_bytes(), "application/pdf"),
        "estimate_pdf": ("sample_estimate.pdf", estimate_pdf.read_bytes(), "application/pdf"),
    }
    create = client.post("/claims", files=files, data={"narrative": "Front-end collision."})
    assert create.status_code == 202, create.text
    claim_id = create.json()["claim_id"]
    payload = client.get(f"/claims/{claim_id}").json()
    assert payload["status"] == "completed"
    result = payload["result"]
    for key in (
        "document_agent",
        "extraction_meta",
        "vision",
        "verifiers",
        "rag",
        "risk",
        "adjudication",
    ):
        assert key in result, f"missing result key {key}"
    assert result["adjudication"]["decision"] == "approve"
    assert result["adjudication"]["cited_clauses"] == ["COL-001"]
    assert result["risk"]["risk_score"] == 0.05


def test_invalid_citation_persists_needs_review(
    client, policy_pdf, estimate_pdf, monkeypatch
):
    """LLM proposes unknown citation → guardrails force needs_review in persisted result."""

    from agents.adjudicator import run_adjudicator

    def llm_bad(_messages):
        return json.dumps(
            {
                "decision": "approve",
                "confidence": 0.99,
                "cited_clauses": ["NOT-RETRIEVED"],
                "risk_flags": [],
                "reasoning_summary": "Hallucinated citation.",
            }
        )

    def real_adjudicator(**kwargs):
        return run_adjudicator(**kwargs, llm_complete=llm_bad)

    monkeypatch.setattr(
        "worker.tasks.run_verifiers",
        lambda *_a, **_k: VerifierOutput(),
    )
    monkeypatch.setattr(
        "worker.tasks.run_fraud_agent",
        lambda *_a, **_k: RiskOutput(flags=[], risk_score=0.0),
    )
    monkeypatch.setattr(
        "worker.tasks.run_rag_agent",
        lambda **_k: RAGOutput(
            retrieved_clauses=[
                RetrievedClause(
                    clause_id="COL-001",
                    text="Collision coverage.",
                    similarity_score=0.88,
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "worker.tasks.run_vision_agent",
        lambda _paths: VisionOutput(
            detections=[],
            severity_tier="minor damage",
            severity_confidence=0.6,
            vqa_answers={},
            low_confidence=False,
        ),
    )
    monkeypatch.setattr("worker.tasks.run_adjudicator", real_adjudicator)

    files = {
        "policy_pdf": ("sample_policy.pdf", policy_pdf.read_bytes(), "application/pdf"),
        "estimate_pdf": ("sample_estimate.pdf", estimate_pdf.read_bytes(), "application/pdf"),
    }
    create = client.post(
        "/claims",
        files=files,
        data={"narrative": "Collision claim for integration guardrail path."},
    )
    claim_id = create.json()["claim_id"]
    payload = client.get(f"/claims/{claim_id}").json()
    assert payload["status"] == "completed"
    adj = payload["result"]["adjudication"]
    assert adj["decision"] == "needs_review"
    assert "NOT-RETRIEVED" not in adj["cited_clauses"]
