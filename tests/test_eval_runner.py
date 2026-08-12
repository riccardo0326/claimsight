"""Offline runner tests for Slice 6 golden eval (oracle / FakeLLM)."""

from __future__ import annotations

from pathlib import Path

from eval.report import report_to_markdown, write_report
from eval.runner import load_manifest, make_oracle_llm, run_case, run_eval

MANIFEST = Path(__file__).resolve().parent.parent / "fixtures" / "golden" / "manifest.jsonl"

# Small stable subset spanning approve / deny / review+fraud
SUBSET_IDS = {
    "g001_collision_approve",
    "g013_intentional_deny",
    "g021_empty_rag_review",
    "g036_weather_mismatch_review",
    "g044_cross_policy_cite_trap",
}


def test_load_manifest_has_fifty_cases():
    cases = load_manifest(MANIFEST)
    assert len(cases) == 50
    assert len({c.claim_id for c in cases}) == 50


def test_oracle_subset_zero_hallucination_and_expected_decisions():
    cases = [c for c in load_manifest(MANIFEST) if c.claim_id in SUBSET_IDS]
    assert len(cases) == len(SUBSET_IDS)

    report = run_eval(cases, mode="fake", model="oracle-fake")
    assert report.metrics.n_cases == len(SUBSET_IDS)
    assert report.metrics.hallucination_rate == 0.0
    assert report.metrics.decision_accuracy == 1.0

    by_id = {c.claim_id: c for c in report.cases}
    assert by_id["g001_collision_approve"].predicted_decision == "approve"
    assert by_id["g013_intentional_deny"].predicted_decision == "deny"
    assert by_id["g021_empty_rag_review"].predicted_decision == "needs_review"
    assert by_id["g036_weather_mismatch_review"].predicted_fraud_flag is True
    assert by_id["g044_cross_policy_cite_trap"].cited_clauses
    assert "OTHER-COL-001" not in by_id["g044_cross_policy_cite_trap"].cited_clauses


def test_hallucinating_llm_forced_to_review_still_zero_hallucination_metric():
    """Guardrails drop illegal cites; post-guardrail hallucination rate stays 0."""
    cases = [c for c in load_manifest(MANIFEST) if c.claim_id == "g001_collision_approve"]
    assert len(cases) == 1
    case = cases[0]

    def bad_llm(_messages):
        import json

        return json.dumps(
            {
                "decision": "approve",
                "confidence": 0.99,
                "cited_clauses": ["OTHER-COL-001", "NOT-REAL"],
                "risk_flags": [],
                "reasoning_summary": "Hallucinated citations on purpose.",
            }
        )

    report_obj, result = run_case(case, llm_complete=bad_llm)
    assert result.hallucinated is False
    assert report_obj.decision == "needs_review"
    assert "OTHER-COL-001" not in report_obj.cited_clauses
    assert "NOT-REAL" not in report_obj.cited_clauses


def test_write_report_roundtrip(tmp_path):
    cases = [c for c in load_manifest(MANIFEST) if c.claim_id == "g001_collision_approve"]
    report = run_eval(cases, mode="fake")
    json_path, md_path = write_report(report, out_dir=tmp_path, basename="smoke")
    assert json_path.exists()
    assert md_path.exists()
    md = report_to_markdown(report)
    assert "Decision accuracy" in md
    assert "g001_collision_approve" in md


def test_make_oracle_llm_returns_json():
    case = next(c for c in load_manifest(MANIFEST) if c.claim_id == "g001_collision_approve")
    raw = make_oracle_llm(case)([])
    assert '"decision": "approve"' in raw or '"decision":"approve"' in raw
