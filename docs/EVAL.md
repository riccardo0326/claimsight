# Slice 6 — Eval harness

Offline-first scoring of **Adjudicator + citation guardrails** against the synthetic golden set in `fixtures/golden/`.

## What is measured

| Metric | Definition | Slice 6 target |
|--------|------------|----------------|
| Citation hallucination rate | Post-guardrail: `cited_clauses ⊈ retrieved clause_ids` | **0%** (hard) |
| Decision accuracy | Exact match vs `ground_truth.decision` | Report baseline (≥85% aspirational; not a CI gate yet) |
| Fraud-flag precision / recall | Material risk flags vs `ground_truth.fraud_flag` | Report baseline only |

Deferred (later slices): RAGAS/faithfulness, Langfuse cost/latency, GitHub Actions §8.3 CI gate, full Celery/HF E2E scoring.

## Eval surface

Each golden case includes canned upstream agent outputs. The harness calls `agents.adjudicator.run_adjudicator` only — it does **not** re-run Document/Vision/Verifiers/RAG/Fraud agents.

## Commands

```bash
# Deterministic oracle LLM (no network) — harness smoke + checked-in sample report
python scripts/run_eval.py --mode fake

# Live frontier LLM (requires OPENAI_API_KEY)
python scripts/run_eval.py --mode live

# Smoke a few cases
python scripts/run_eval.py --mode fake --limit 5
```

Reports land in `eval/reports/latest.json` and `eval/reports/latest.md`.

## Pytest

Default suite covers metric math and a small offline runner subset (`tests/test_eval_*.py`). Full live golden runs stay opt-in via `pytest -m live_llm` / the script above.

See `docs/DECISIONS.md` Slice 6 and `fixtures/golden/README.md`.
