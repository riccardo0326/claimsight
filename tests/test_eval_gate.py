"""Unit tests for Slice 7 eval CI gates."""

from __future__ import annotations

import json
from pathlib import Path

from eval.gate import check_gates, load_baseline
from eval.schema import EvalMetrics, EvalReport


def _metrics(
    *,
    accuracy: float = 1.0,
    hallucination: float = 0.0,
    n: int = 50,
) -> EvalMetrics:
    return EvalMetrics(
        n_cases=n,
        decision_accuracy=accuracy,
        hallucination_rate=hallucination,
        fraud_precision=1.0,
        fraud_recall=1.0,
        fraud_true_positives=8,
        fraud_false_positives=0,
        fraud_false_negatives=0,
        fraud_true_negatives=42,
    )


def test_check_gates_perfect_pass():
    baseline = _metrics()
    current = _metrics()
    assert check_gates(current, baseline) == []


def test_check_gates_hallucination_fails():
    baseline = _metrics()
    current = _metrics(hallucination=0.02)
    failures = check_gates(current, baseline)
    assert len(failures) == 1
    assert "hallucination_rate" in failures[0]


def test_check_gates_accuracy_drop_003_fails():
    baseline = _metrics(accuracy=1.0)
    current = _metrics(accuracy=0.97)
    failures = check_gates(current, baseline)
    assert len(failures) == 1
    assert "decision_accuracy" in failures[0]


def test_check_gates_accuracy_drop_001_passes():
    baseline = _metrics(accuracy=1.0)
    current = _metrics(accuracy=0.99)
    assert check_gates(current, baseline) == []


def test_check_gates_both_can_fail():
    baseline = _metrics(accuracy=1.0)
    current = _metrics(accuracy=0.90, hallucination=0.1)
    failures = check_gates(current, baseline)
    assert len(failures) == 2


def test_load_baseline_from_report_json(tmp_path: Path):
    report = EvalReport(
        mode="fake",
        model="oracle-fake",
        metrics=_metrics(),
        cases=[],
    )
    path = tmp_path / "baseline.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    loaded = load_baseline(path)
    assert loaded.decision_accuracy == 1.0
    assert loaded.hallucination_rate == 0.0


def test_load_baseline_from_bare_metrics(tmp_path: Path):
    path = tmp_path / "metrics.json"
    path.write_text(_metrics().model_dump_json(), encoding="utf-8")
    loaded = load_baseline(path)
    assert loaded.n_cases == 50


def test_checked_in_baseline_loads():
    root = Path(__file__).resolve().parent.parent
    baseline_path = root / "eval" / "reports" / "baseline_fake.json"
    assert baseline_path.exists()
    metrics = load_baseline(baseline_path)
    assert metrics.n_cases == 50
    assert metrics.hallucination_rate == 0.0
    # Oracle fake baseline should be perfect accuracy.
    assert metrics.decision_accuracy == 1.0
