"""CI gate checks for Slice 7 golden eval (partial PROJECT_SPEC §8.3)."""

from __future__ import annotations

import json
from pathlib import Path

from eval.schema import EvalMetrics, EvalReport

DEFAULT_MAX_HALLUCINATION = 0.0
DEFAULT_MAX_ACCURACY_DROP = 0.02


def load_baseline(path: Path | str) -> EvalMetrics:
    """Load metrics from a checked-in EvalReport JSON (or bare EvalMetrics)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "metrics" in data:
        return EvalReport.model_validate(data).metrics
    return EvalMetrics.model_validate(data)


def check_gates(
    current: EvalMetrics,
    baseline: EvalMetrics,
    *,
    max_hallucination: float = DEFAULT_MAX_HALLUCINATION,
    max_accuracy_drop: float = DEFAULT_MAX_ACCURACY_DROP,
) -> list[str]:
    """Return human-readable failure reasons (empty list = pass).

    Faithfulness regression (§8.3) is deferred until RAGAS/LLM-judge exists.
    """
    failures: list[str] = []

    if current.hallucination_rate > max_hallucination:
        failures.append(
            f"hallucination_rate {current.hallucination_rate:.4f} exceeds "
            f"max {max_hallucination:.4f}"
        )

    drop = baseline.decision_accuracy - current.decision_accuracy
    if drop > max_accuracy_drop:
        failures.append(
            f"decision_accuracy dropped {drop:.4f} "
            f"(baseline {baseline.decision_accuracy:.4f} → "
            f"current {current.decision_accuracy:.4f}; "
            f"max allowed drop {max_accuracy_drop:.4f})"
        )

    return failures
