# Eval harness + CI gate (Slices 6–7)

Offline-first scoring of **Adjudicator + citation guardrails** against the synthetic golden set in `fixtures/golden/`, with a GitHub Actions gate on PRs and `main`.

## What is measured

| Metric | Definition | Gate |
|--------|------------|------|
| Citation hallucination rate | Post-guardrail: `cited_clauses ⊈ retrieved clause_ids` | **Fail if > 0%** |
| Decision accuracy | Exact match vs `ground_truth.decision` | **Fail if drop > 0.02** vs `eval/reports/baseline_fake.json` |
| Fraud-flag precision / recall | Material risk flags vs `ground_truth.fraud_flag` | Report only (not gated) |

Deferred: RAGAS/faithfulness CI delta, live OpenAI eval in CI, full Celery/HF E2E scoring, golden ≥150.
Adjudicator token cost/latency: see [COST_ROUTING.md](COST_ROUTING.md) and Langfuse (Slice 8) — not CI-gated.

**Accuracy caveat:** `--mode fake` uses an oracle stub keyed to ground truth, so accuracy is usually ~1.0. The accuracy gate protects harness/GT consistency; it is not a live model quality gate (see DECISIONS D36).

## Eval surface

Each golden case includes canned upstream agent outputs. The harness calls `agents.adjudicator.run_adjudicator` only — it does **not** re-run Document/Vision/Verifiers/RAG/Fraud agents.

## Commands

```bash
# Deterministic oracle LLM (no network)
python scripts/run_eval.py --mode fake

# Same + CI gates vs checked-in baseline (exit 1 on failure)
python scripts/run_eval.py --mode fake --gate

# Live frontier LLM (requires OPENAI_API_KEY) — not used in CI
python scripts/run_eval.py --mode live

# Smoke a few cases
python scripts/run_eval.py --mode fake --limit 5
```

Reports land in `eval/reports/latest.json` and `eval/reports/latest.md`.

### Refreshing the baseline

When you intentionally change golden labels or oracle expectations:

```bash
python scripts/run_eval.py --mode fake
cp eval/reports/latest.json eval/reports/baseline_fake.json
```

Commit the updated `baseline_fake.json` in the same PR and explain why.

## CI

Workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

- Python 3.12 on `ubuntu-latest`
- `pytest` (default markers: no HF / live APIs / live LLM)
- `python scripts/run_eval.py --mode fake --gate`
- Triggers: all `pull_request`s and pushes to `main`
- No repo secrets required

## Pytest

Default suite covers metric math, runner subset, and gate unit tests (`tests/test_eval_*.py`). Full live golden runs stay opt-in via `pytest -m live_llm` / `--mode live`.

See `docs/DECISIONS.md` Slices 6–7 and `fixtures/golden/README.md`.
