"""Live / manual Adjudicator verification (OpenAI frontier LLM).

Not part of the default pytest suite. Requires OPENAI_API_KEY.

  python scripts/verify_adjudicator_live.py
  pytest -m live_llm
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.adjudicator import run_adjudicator  # noqa: E402
from agents.adjudicator_guardrails import apply_guardrails  # noqa: E402
from agents.schemas import (  # noqa: E402
    ClaimReport,
    DocumentOutput,
    RAGOutput,
    RetrievedClause,
    RiskOutput,
    VerifierOutput,
    VisionOutput,
)

RESULTS_PATH = ROOT / "docs" / "ADJUDICATOR_LIVE_VERIFY.md"


def _fixture_evidence() -> dict:
    return {
        "narrative": (
            "The insured vehicle was damaged in a front-end collision with another car. "
            "Please confirm collision coverage and repair payment after deductible."
        ),
        "document": DocumentOutput(
            policy_id="POL-2024-0098213",
            coverage_limits={"collision": 50000.0},
            deductible=500.0,
            vin="1HGCM82633A004352",
            incident_date=date(2024, 3, 14),
            line_items=[{"description": "Front bumper replacement", "cost": 850.0}],
        ),
        "extraction_meta": {
            "confidences": {"policy_id": 0.99, "coverage_limits.collision": 0.95},
            "low_confidence_fields": [],
            "min_confidence": 0.5,
        },
        "vision": VisionOutput(
            detections=[
                {
                    "label": "damaged car",
                    "confidence": 0.72,
                    "image_path": "fixtures/images/synthetic_0.jpg",
                }
            ],
            severity_tier="moderate damage",
            severity_confidence=0.65,
            vqa_answers={"Is the airbag deployed?": "no"},
            low_confidence=False,
        ),
        "rag": RAGOutput(
            retrieved_clauses=[
                RetrievedClause(
                    clause_id="COL-001",
                    text=(
                        "Collision coverage pays for direct and accidental loss to your "
                        "covered auto caused by collision with another object or by upset "
                        "of the auto, subject to the deductible shown on the declarations page."
                    ),
                    similarity_score=0.91,
                ),
                RetrievedClause(
                    clause_id="COL-002",
                    text=(
                        "Under collision coverage, we will pay the lesser of the actual "
                        "cash value of the damaged property or the amount necessary to "
                        "repair or replace it with other property of like kind and quality."
                    ),
                    similarity_score=0.84,
                ),
            ]
        ),
        "verifiers": VerifierOutput(
            make="HONDA",
            model="Accord",
            model_year=2003,
            sources_failed=[],
        ),
        "risk": RiskOutput(flags=[], risk_score=0.05),
    }


def test_deterministic_invalid_citation_rejected():
    """Guardrail check does not require a live model."""
    evidence = _fixture_evidence()
    proposed = ClaimReport(
        decision="approve",
        confidence=0.99,
        cited_clauses=["NOT-IN-RETRIEVED-SET"],
        risk_flags=[],
        reasoning_summary="Forged citation for live-verify guardrail check.",
    )
    out = apply_guardrails(
        proposed,
        rag=evidence["rag"],
        vision=evidence["vision"],
        verifiers=evidence["verifiers"],
        risk=evidence["risk"],
        extraction_meta=evidence["extraction_meta"],
    )
    assert out.decision == "needs_review"
    assert "NOT-IN-RETRIEVED-SET" not in out.cited_clauses


@pytest.mark.live_llm
def test_live_openai_adjudicator_roundtrip():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    evidence = _fixture_evidence()
    report = run_adjudicator(
        narrative=evidence["narrative"],
        document=evidence["document"],
        extraction_meta=evidence["extraction_meta"],
        vision=evidence["vision"],
        rag=evidence["rag"],
        verifiers=evidence["verifiers"],
        risk=evidence["risk"],
    )
    assert isinstance(report, ClaimReport)
    assert report.decision in {"approve", "deny", "needs_review"}
    retrieved = {c.clause_id for c in evidence["rag"].retrieved_clauses}
    assert set(report.cited_clauses) <= retrieved
    assert 0.0 <= report.confidence <= 1.0


def _write_results_md(
    *,
    live_ran: bool,
    live_ok: bool | None,
    report: ClaimReport | None,
    error: str | None,
    guardrail_ok: bool,
) -> None:
    from api.config import get_settings

    settings = get_settings()
    lines = [
        "# Slice 5 — Adjudicator live verification",
        "",
        "Manual procedure (network + `OPENAI_API_KEY` required). Not part of default `pytest`.",
        "",
        "## Prerequisites",
        "",
        "- Python 3.12 venv with project deps",
        "- `OPENAI_API_KEY` set in the environment (never commit the key)",
        f"- Model: `{settings.adjudicator_model}` via `{settings.adjudicator_base_url}`",
        "",
        "## Run",
        "",
        "```bash",
        "python scripts/verify_adjudicator_live.py",
        "",
        "# Optional pytest marker (excluded from default suite)",
        "pytest -m live_llm",
        "```",
        "",
        "## Checklist",
        "",
        "1. Real model invocation succeeds",
        "2. Structured ClaimReport can be produced",
        "3. Valid citations are a subset of provided RAG clause_ids",
        "4. Deterministic validation accepts a valid response",
        "5. Invalid citation fixture is rejected deterministically (no live model needed)",
        "",
        f"## Results log ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Provider | OpenAI Chat Completions (`httpx`) |",
        f"| Model | `{settings.adjudicator_model}` |",
        f"| Deterministic invalid-citation guardrail | {'PASS' if guardrail_ok else 'FAIL'} |",
    ]
    if not live_ran:
        lines.append(
            "| Live OpenAI call | **NOT RUN** — `OPENAI_API_KEY` missing or skipped |"
        )
        lines.append("")
        lines.append(
            "**Limitation:** Live frontier verification was not executed in this environment. "
            "Offline unit/integration tests cover guardrails and mocked LLM paths."
        )
    elif live_ok and report is not None:
        lines.append("| Live OpenAI call | PASS |")
        lines.append(f"| Decision | `{report.decision}` |")
        lines.append(f"| Confidence | `{report.confidence}` |")
        lines.append(f"| Cited clauses | `{report.cited_clauses}` |")
        lines.append("| Citation subset check | PASS |")
        lines.append("")
        lines.append("### Model report (truncated)")
        lines.append("")
        lines.append("```json")
        dump = report.model_dump(mode="json")
        dump["reasoning_summary"] = (dump.get("reasoning_summary") or "")[:800]
        lines.append(json.dumps(dump, indent=2))
        lines.append("```")
    else:
        lines.append(f"| Live OpenAI call | FAIL — {error or 'unknown error'} |")
        lines.append("")
        lines.append(
            "**Limitation:** Live call failed. Offline suite still covers deterministic "
            "guardrails; re-run when credentials/network allow."
        )
    lines.append("")
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")


def main() -> int:
    # 1) Deterministic guardrail (always)
    try:
        test_deterministic_invalid_citation_rejected()
        guardrail_ok = True
        print("PASS deterministic invalid-citation guardrail")
    except AssertionError as exc:
        guardrail_ok = False
        print(f"FAIL deterministic guardrail: {exc}")

    # 2) Live LLM if key present
    live_ran = False
    live_ok: bool | None = None
    report: ClaimReport | None = None
    error: str | None = None

    if os.environ.get("OPENAI_API_KEY"):
        live_ran = True
        try:
            from api.config import get_settings

            get_settings.cache_clear()
            evidence = _fixture_evidence()
            report = run_adjudicator(
                narrative=evidence["narrative"],
                document=evidence["document"],
                extraction_meta=evidence["extraction_meta"],
                vision=evidence["vision"],
                rag=evidence["rag"],
                verifiers=evidence["verifiers"],
                risk=evidence["risk"],
            )
            retrieved = {c.clause_id for c in evidence["rag"].retrieved_clauses}
            assert set(report.cited_clauses) <= retrieved
            assert report.decision in {"approve", "deny", "needs_review"}
            live_ok = True
            print(
                f"PASS live adjudicator decision={report.decision} "
                f"cites={report.cited_clauses}"
            )
        except Exception as exc:  # noqa: BLE001
            live_ok = False
            error = str(exc)
            print(f"FAIL live adjudicator: {exc}")
    else:
        print("SKIP live OpenAI call (OPENAI_API_KEY not set)")

    _write_results_md(
        live_ran=live_ran,
        live_ok=live_ok,
        report=report,
        error=error,
        guardrail_ok=guardrail_ok,
    )

    if not guardrail_ok:
        return 1
    if live_ran and not live_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
