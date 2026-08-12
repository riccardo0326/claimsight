# ClaimSight golden dataset (Slice 6)

Synthetic labeled claims for the Adjudicator eval harness. **No real customer PII.**

## Layout

| Path | Purpose |
|------|---------|
| `manifest.jsonl` | One `GoldenCase` JSON object per line (~50 cases) |
| `build_manifest.py` | Regenerates `manifest.jsonl` from templates |
| `README.md` | This file |

Shared PDFs/images are reused from `fixtures/` (sample policy/estimate, synthetic photos). Each case carries a canned **upstream** snapshot (`document_agent`, `extraction_meta`, `vision`, `verifiers`, `rag`, `risk`) so eval scores **Adjudicator + guardrails** without Celery/HF/live NHTSA.

## Schema (extends PROJECT_SPEC §8.1)

```json
{
  "claim_id": "g001_collision_approve",
  "narrative": "...",
  "policy_pdf": "fixtures/sample_policy.pdf",
  "estimate_pdf": "fixtures/sample_estimate.pdf",
  "images": [],
  "incident_location": null,
  "upstream": { "document_agent": {}, "extraction_meta": {}, "vision": null, "verifiers": {}, "rag": {}, "risk": {} },
  "ground_truth": {
    "decision": "approve | deny | needs_review",
    "clause_ids": ["COL-001"],
    "fraud_flag": false
  },
  "notes": "..."
}
```

- `clause_ids` is a **list** (multi-cite).
- `fraud_flag` is true when any material risk flag is expected:
  `weather mismatch` | `possible staged damage` | `inconsistent claim`.

## Case mix (~50)

| Bucket | Count | Intent |
|--------|------:|--------|
| Approve | 12 | Clear coverage + legal citations |
| Deny | 8 | Exclusion evidence retrieved |
| Needs review (evidence gaps) | 15 | Empty RAG, sources_failed, low-confidence extraction, ambiguity |
| Fraud-signal → review | 8 | Material risk flags; fraud P/R baseline |
| Edge | 7 | Cross-policy cite trap, empty vision, multi-cite, procedure-only RAG |

## Regenerate

```bash
python fixtures/golden/build_manifest.py
```

## Run eval

```bash
python scripts/run_eval.py --mode fake
# optional live Adjudicator:
python scripts/run_eval.py --mode live
```

See `docs/EVAL.md` and `docs/DECISIONS.md` (Slice 6).
