"""Slice 9 LangGraph topology and invoke tests (offline, mocked agents)."""

from __future__ import annotations

from langgraph.graph import END, START

from agents.schemas import (
    ClaimReport,
    DocumentOutput,
    RAGOutput,
    RetrievedClause,
    RiskOutput,
    VerifierOutput,
    VisionOutput,
)
from graph.claim_graph import (
    get_claim_graph,
    graph_edge_pairs,
    invoke_claim_graph,
    result_from_state,
)
from graph.state import ClaimState


def _initial(**overrides) -> ClaimState:
    state: ClaimState = {
        "claim_id": "00000000-0000-0000-0000-000000000001",
        "narrative": "Front-end collision.",
        "incident_location": "Washington, DC",
        "policy_pdf": "fixtures/sample_policy.pdf",
        "estimate_pdf": "fixtures/sample_estimate.pdf",
        "image_paths": [],
    }
    state.update(overrides)
    return state


def test_graph_edges_honest_dag():
    pairs = graph_edge_pairs()
    start = str(START)
    end = str(END)
    assert (start, "document") in pairs
    assert (start, "vision") in pairs
    assert ("document", "rag") in pairs
    assert ("document", "verifiers") in pairs
    assert ("verifiers", "fraud_risk") in pairs
    assert ("vision", "adjudicator") in pairs
    assert ("rag", "adjudicator") in pairs
    assert ("fraud_risk", "adjudicator") in pairs
    assert ("adjudicator", end) in pairs
    # Sequential Celery order must not be encoded as edges.
    assert ("document", "vision") not in pairs
    assert ("vision", "verifiers") not in pairs
    assert ("verifiers", "rag") not in pairs
    assert ("rag", "fraud_risk") not in pairs
    # Verifiers cannot start with Document — they need VIN/date.
    assert (start, "verifiers") not in pairs


def test_invoke_skip_vision_and_persist_contract(monkeypatch):
    doc = DocumentOutput(policy_id="POL-1", vin="1HGCM82633A004352")
    meta = {"confidences": {}, "low_confidence_fields": [], "min_confidence": 0.5}

    monkeypatch.setattr(
        "graph.nodes.run_document_agent",
        lambda *_a, **_k: (doc, meta),
    )
    monkeypatch.setattr("graph.nodes.run_vision_agent", lambda _paths: None)
    monkeypatch.setattr(
        "graph.nodes.run_verifiers",
        lambda *_a, **_k: VerifierOutput(make="HONDA", model="Accord", model_year=2003),
    )
    monkeypatch.setattr(
        "graph.nodes.run_rag_agent",
        lambda **_k: RAGOutput(
            retrieved_clauses=[
                RetrievedClause(clause_id="COL-001", text="Collision.", similarity_score=0.9)
            ]
        ),
    )
    monkeypatch.setattr(
        "graph.nodes.run_fraud_agent",
        lambda *_a, **_k: RiskOutput(flags=[], risk_score=0.05),
    )
    monkeypatch.setattr(
        "graph.nodes.run_adjudicator",
        lambda **_k: ClaimReport(
            decision="approve",
            confidence=0.7,
            cited_clauses=["COL-001"],
            risk_flags=[],
            reasoning_summary="Graph stub approve.",
        ),
    )

    final = invoke_claim_graph(_initial(), db=None, max_concurrency=1)
    assert final["vision"] is None
    assert final["document"].policy_id == "POL-1"
    assert final["adjudication"].decision == "approve"
    result = result_from_state(final)
    for key in (
        "document_agent",
        "extraction_meta",
        "vision",
        "verifiers",
        "rag",
        "risk",
        "adjudication",
    ):
        assert key in result
    assert result["vision"] is None
    assert result["adjudication"]["cited_clauses"] == ["COL-001"]
    assert result["risk"]["risk_score"] == 0.05


def test_invoke_with_photos_keeps_vision(monkeypatch):
    vision = VisionOutput(
        detections=[],
        severity_tier="moderate damage",
        severity_confidence=0.6,
        vqa_answers={},
        low_confidence=False,
    )
    monkeypatch.setattr(
        "graph.nodes.run_document_agent",
        lambda *_a, **_k: (DocumentOutput(policy_id="POL-1"), {"low_confidence_fields": []}),
    )
    monkeypatch.setattr("graph.nodes.run_vision_agent", lambda paths: vision)
    monkeypatch.setattr(
        "graph.nodes.run_verifiers",
        lambda *_a, **_k: VerifierOutput(),
    )
    monkeypatch.setattr(
        "graph.nodes.run_rag_agent",
        lambda **_k: RAGOutput(retrieved_clauses=[]),
    )
    monkeypatch.setattr(
        "graph.nodes.run_fraud_agent",
        lambda *_a, **_k: RiskOutput(flags=[], risk_score=0.0),
    )
    monkeypatch.setattr(
        "graph.nodes.run_adjudicator",
        lambda **_k: ClaimReport(
            decision="needs_review",
            confidence=0.4,
            cited_clauses=[],
            risk_flags=[],
            reasoning_summary="Empty RAG.",
        ),
    )

    final = invoke_claim_graph(
        _initial(image_paths=["fixtures/images/synthetic_front.jpg"]),
        db=None,
        max_concurrency=1,
    )
    assert final["vision"] is vision
    assert result_from_state(final)["vision"]["severity_tier"] == "moderate damage"


def test_compiled_graph_is_cached():
    assert get_claim_graph() is get_claim_graph()


def test_graph_nodes_emit_langfuse_span_names(monkeypatch):
    from agents.observability import set_client_override
    from test_observability import FakeLangfuse

    fake = FakeLangfuse()
    set_client_override(fake)
    try:
        monkeypatch.setattr(
            "graph.nodes.run_document_agent",
            lambda *_a, **_k: (DocumentOutput(policy_id="POL-1"), {}),
        )
        monkeypatch.setattr("graph.nodes.run_vision_agent", lambda _paths: None)
        monkeypatch.setattr(
            "graph.nodes.run_verifiers",
            lambda *_a, **_k: VerifierOutput(),
        )
        monkeypatch.setattr(
            "graph.nodes.run_rag_agent",
            lambda **_k: RAGOutput(retrieved_clauses=[]),
        )
        monkeypatch.setattr(
            "graph.nodes.run_fraud_agent",
            lambda *_a, **_k: RiskOutput(flags=[], risk_score=0.0),
        )
        monkeypatch.setattr(
            "graph.nodes.run_adjudicator",
            lambda **_k: ClaimReport(
                decision="needs_review",
                confidence=0.4,
                cited_clauses=[],
                risk_flags=[],
                reasoning_summary="span test",
            ),
        )
        invoke_claim_graph(_initial(), db=None, max_concurrency=1)
        for name in (
            "document",
            "vision",
            "verifiers",
            "rag",
            "fraud_risk",
            "adjudicator",
        ):
            assert f"span:{name}" in fake.events
    finally:
        set_client_override(None)
