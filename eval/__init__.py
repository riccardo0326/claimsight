"""ClaimSight golden-set eval harness (Slice 6+) with CI gates (Slice 7)."""

from __future__ import annotations

from eval.gate import check_gates, load_baseline
from eval.metrics import (
    CaseScore,
    aggregate_metrics,
    predicted_fraud_flag,
    score_case,
)
from eval.report import write_report
from eval.runner import load_manifest, run_eval
from eval.schema import EvalReport, GoldenCase, GroundTruth, UpstreamSnapshot

__all__ = [
    "CaseScore",
    "EvalReport",
    "GoldenCase",
    "GroundTruth",
    "UpstreamSnapshot",
    "aggregate_metrics",
    "check_gates",
    "load_baseline",
    "load_manifest",
    "predicted_fraud_flag",
    "run_eval",
    "score_case",
    "write_report",
]
