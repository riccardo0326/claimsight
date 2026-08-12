"""Load golden cases and run Adjudicator + guardrails for eval."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from agents.adjudicator import run_adjudicator
from agents.schemas import ClaimReport
from eval.metrics import aggregate_metrics, predicted_fraud_flag, score_case
from eval.schema import CaseResult, EvalReport, GoldenCase


def load_manifest(path: Path | str) -> list[GoldenCase]:
    """Load JSONL golden manifest (one GoldenCase object per line)."""
    p = Path(path)
    cases: list[GoldenCase] = []
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            cases.append(GoldenCase.model_validate_json(text))
        except Exception as exc:  # noqa: BLE001 — surface line number
            raise ValueError(f"Invalid golden case at {p}:{line_no}: {exc}") from exc
    return cases


def make_oracle_llm(case: GoldenCase) -> Callable[[list[dict[str, str]]], str]:
    """Deterministic stub that proposes ground-truth decision with legal citations.

    Used for offline smoke / harness validation. Citations are intersected with
    retrieved clause_ids so the proposal is guardrail-legal when GT is consistent.
    """

    retrieved = {c.clause_id for c in case.upstream.rag.retrieved_clauses}
    preferred = [c for c in case.ground_truth.clause_ids if c in retrieved]
    if case.ground_truth.decision == "approve" and not preferred and retrieved:
        preferred = [next(iter(sorted(retrieved)))]
    if case.ground_truth.decision == "deny" and not preferred:
        # Prefer an exclusion-looking id if present; else any retrieved.
        exclusions = [c for c in retrieved if c.startswith("EXC-") or "EXC" in c]
        preferred = exclusions[:1] or (sorted(retrieved)[:1] if retrieved else [])

    payload = {
        "decision": case.ground_truth.decision,
        "confidence": 0.5,
        "cited_clauses": preferred,
        "risk_flags": [],
        "reasoning_summary": (
            f"Oracle fake LLM for {case.claim_id}: "
            f"propose {case.ground_truth.decision}."
        ),
    }

    def _complete(_messages: list[dict[str, str]]) -> str:
        return json.dumps(payload)

    return _complete


def run_case(
    case: GoldenCase,
    *,
    llm_complete: Callable[[list[dict[str, str]]], str] | None = None,
) -> tuple[ClaimReport, CaseResult]:
    """Run Adjudicator+guardrails for one golden case and score it."""
    complete = llm_complete
    report = run_adjudicator(
        narrative=case.narrative,
        document=case.upstream.document_agent,
        extraction_meta=case.upstream.extraction_meta,
        vision=case.upstream.vision,
        rag=case.upstream.rag,
        verifiers=case.upstream.verifiers,
        risk=case.upstream.risk,
        llm_complete=complete,
    )
    scored = score_case(case, report)
    result = CaseResult(
        claim_id=case.claim_id,
        predicted_decision=report.decision,
        ground_truth_decision=case.ground_truth.decision,
        decision_match=scored.decision_match,
        hallucinated=scored.hallucinated,
        predicted_fraud_flag=predicted_fraud_flag(report),
        ground_truth_fraud_flag=case.ground_truth.fraud_flag,
        cited_clauses=list(report.cited_clauses),
        confidence=report.confidence,
        reasoning_summary=report.reasoning_summary,
    )
    return report, result


def run_eval(
    cases: list[GoldenCase],
    *,
    mode: str,
    llm_complete_factory: (
        Callable[[GoldenCase], Callable[[list[dict[str, str]]], str]] | None
    ) = None,
    model: str | None = None,
    prompt_version: str = "prompts/adjudicator_v1.md",
) -> EvalReport:
    """Score a list of golden cases.

    mode:
      - fake: use oracle LLM from make_oracle_llm (default factory)
      - live: llm_complete_factory None → run_adjudicator uses real OpenAI client
      - custom: pass llm_complete_factory
    """
    results: list[CaseResult] = []
    scores = []

    for case in cases:
        if mode == "fake":
            factory = llm_complete_factory or make_oracle_llm
            llm = factory(case)
        elif llm_complete_factory is not None:
            llm = llm_complete_factory(case)
        else:
            llm = None  # live path inside run_adjudicator

        _report, result = run_case(case, llm_complete=llm)
        results.append(result)
        scores.append(score_case(case, _report))

    return EvalReport(
        mode=mode,
        prompt_version=prompt_version,
        model=model,
        metrics=aggregate_metrics(scores),
        cases=results,
    )
