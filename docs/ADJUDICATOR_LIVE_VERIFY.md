# Slice 5 — Adjudicator live verification

Manual procedure (network + `OPENAI_API_KEY` required). Not part of default `pytest`.

## Prerequisites

- Python 3.12 venv with project deps
- `OPENAI_API_KEY` set in the environment (never commit the key)
- Model: `gpt-4o` via `https://api.openai.com/v1`

## Run

```bash
python scripts/verify_adjudicator_live.py

# Optional pytest marker (excluded from default suite)
pytest -m live_llm
```

## Checklist

1. Real model invocation succeeds
2. Structured ClaimReport can be produced
3. Valid citations are a subset of provided RAG clause_ids
4. Deterministic validation accepts a valid response
5. Invalid citation fixture is rejected deterministically (no live model needed)

## Results log (2026-08-12 09:35:25 UTC)

| Check | Result |
|---|---|
| Provider | OpenAI Chat Completions (`httpx`) |
| Model | `gpt-4o` |
| Deterministic invalid-citation guardrail | PASS |
| Live OpenAI call | PASS |
| Decision | `approve` |
| Confidence | `0.8` |
| Cited clauses | `['COL-001', 'COL-002']` |
| Citation subset check | PASS |

### Model report (truncated)

```json
{
  "decision": "approve",
  "confidence": 0.8,
  "cited_clauses": [
    "COL-001",
    "COL-002"
  ],
  "risk_flags": [],
  "reasoning_summary": "FACT/EVIDENCE: The claim involves a front-end collision with another car, which is covered under the collision coverage as per clause COL-001. The policy provides collision coverage with a limit of $50,000 and a deductible of $500. The vision analysis confirms moderate damage to the vehicle, consistent with the claim of a collision. INFERENCE: The cost of repair ($850) is less than the coverage limit and will be subject to the deductible. UNKNOWN/MISSING: There is no weather verification, but this does not impact the collision coverage decision. The risk score is low, and there are no risk flags. Therefore, the claim is approved based on the available evidence and policy terms."
}
```

