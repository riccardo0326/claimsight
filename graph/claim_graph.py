"""Compile and invoke the ClaimSight LangGraph (Slice 9)."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from api.config import get_settings
from graph.nodes import (
    adjudicator_node,
    document_node,
    fraud_risk_node,
    rag_node,
    verifiers_node,
    vision_node,
)
from graph.state import ClaimState

_compiled = None


def build_claim_graph() -> StateGraph:
    """Uncompiled graph: Vision ∥ Document; Verifiers/RAG after Document; join at Adjudicator."""
    builder = StateGraph(ClaimState)
    builder.add_node("document", document_node)
    builder.add_node("vision", vision_node)
    builder.add_node("verifiers", verifiers_node)
    builder.add_node("rag", rag_node)
    builder.add_node("fraud_risk", fraud_risk_node)
    try:
        builder.add_node("adjudicator", adjudicator_node, defer=True)
    except TypeError:
        builder.add_node("adjudicator", adjudicator_node)

    builder.add_edge(START, "document")
    builder.add_edge(START, "vision")
    builder.add_edge("document", "rag")
    builder.add_edge("document", "verifiers")
    builder.add_edge("verifiers", "fraud_risk")
    builder.add_edge("vision", "adjudicator")
    builder.add_edge("rag", "adjudicator")
    builder.add_edge("fraud_risk", "adjudicator")
    builder.add_edge("adjudicator", END)
    return builder


def get_claim_graph():
    """Return a process-wide compiled graph (no checkpointer)."""
    global _compiled
    if _compiled is None:
        _compiled = build_claim_graph().compile()
    return _compiled


def graph_edge_pairs(compiled=None) -> set[tuple[str, str]]:
    """Normalized (source, target) pairs from the compiled drawable graph."""
    graph = compiled if compiled is not None else get_claim_graph()
    drawable = graph.get_graph()
    pairs: set[tuple[str, str]] = set()
    for edge in drawable.edges:
        src = getattr(edge, "source", None)
        tgt = getattr(edge, "target", None)
        if src is None and isinstance(edge, (tuple, list)) and len(edge) >= 2:
            src, tgt = edge[0], edge[1]
        if src is None or tgt is None:
            continue
        pairs.add((str(src), str(tgt)))
    return pairs


def invoke_claim_graph(
    state: ClaimState,
    *,
    db: Any,
    max_concurrency: int | None = None,
) -> ClaimState:
    """Run the compiled graph. `db` is injected via RunnableConfig, not state."""
    graph = get_claim_graph()
    settings = get_settings()
    concurrency = (
        max_concurrency
        if max_concurrency is not None
        else settings.graph_max_concurrency
    )
    config: dict[str, Any] = {"configurable": {"db": db}}
    if concurrency and concurrency > 0:
        config["max_concurrency"] = concurrency
    return graph.invoke(state, config)


def result_from_state(state: ClaimState) -> dict[str, Any]:
    """Map ClaimState onto the persisted claim.result contract (D22/D25)."""
    document = state["document"]
    vision = state.get("vision")
    return {
        "document_agent": document.model_dump(mode="json"),
        "extraction_meta": state["extraction_meta"],
        "vision": vision.model_dump(mode="json") if vision else None,
        "verifiers": state["verifiers"].model_dump(mode="json"),
        "rag": state["rag"].model_dump(mode="json"),
        "risk": state["risk"].model_dump(mode="json"),
        "adjudication": state["adjudication"].model_dump(mode="json"),
    }
