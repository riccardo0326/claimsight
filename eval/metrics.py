"""Pure metric helpers for Slice 6 golden eval."""

from __future__ import annotations

from dataclasses import dataclass

from agents.adjudicator_guardrails import MATERIAL_RISK_FLAG_TYPES, citations_valid
from agents.schemas import ClaimReport, RAGOutput
from eval.schema import EvalMetrics, GoldenCase


@dataclass(frozen=True)
class CaseScore:
    claim_id: str
    decision_match: bool
    hallucinated: bool
    predicted_fraud: bool
    ground_truth_fraud: bool


def predicted_fraud_flag(report: ClaimReport) -> bool:
    """True when any material risk flag is present on the ClaimReport."""
    return any(f.flag_type in MATERIAL_RISK_FLAG_TYPES for f in report.risk_flags)


def is_hallucinated(report: ClaimReport, rag: RAGOutput) -> bool:
    """Post-guardrail citation hallucination: cited ⊈ retrieved."""
    return not citations_valid(report.cited_clauses, rag)


def score_case(case: GoldenCase, report: ClaimReport) -> CaseScore:
    return CaseScore(
        claim_id=case.claim_id,
        decision_match=report.decision == case.ground_truth.decision,
        hallucinated=is_hallucinated(report, case.upstream.rag),
        predicted_fraud=predicted_fraud_flag(report),
        ground_truth_fraud=case.ground_truth.fraud_flag,
    )


def aggregate_metrics(scores: list[CaseScore]) -> EvalMetrics:
    n = len(scores)
    if n == 0:
        return EvalMetrics(
            n_cases=0,
            decision_accuracy=0.0,
            hallucination_rate=0.0,
            fraud_precision=None,
            fraud_recall=None,
        )

    decision_hits = sum(1 for s in scores if s.decision_match)
    hallucinations = sum(1 for s in scores if s.hallucinated)

    tp = sum(1 for s in scores if s.predicted_fraud and s.ground_truth_fraud)
    fp = sum(1 for s in scores if s.predicted_fraud and not s.ground_truth_fraud)
    fn = sum(1 for s in scores if not s.predicted_fraud and s.ground_truth_fraud)
    tn = sum(1 for s in scores if not s.predicted_fraud and not s.ground_truth_fraud)

    precision = (tp / (tp + fp)) if (tp + fp) else None
    recall = (tp / (tp + fn)) if (tp + fn) else None

    return EvalMetrics(
        n_cases=n,
        decision_accuracy=decision_hits / n,
        hallucination_rate=hallucinations / n,
        fraud_precision=precision,
        fraud_recall=recall,
        fraud_true_positives=tp,
        fraud_false_positives=fp,
        fraud_false_negatives=fn,
        fraud_true_negatives=tn,
    )
