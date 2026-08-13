# Langfuse live verification (Slice 8)

Optional check that a completed claim produces a searchable Langfuse trace.

## Prerequisites

- Stack up (`docker compose up --build` or local API + worker)
- `OPENAI_API_KEY` set (Adjudicator)
- `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` set (see `.env.example`)
- Optional: `LANGFUSE_HOST` (default `https://cloud.langfuse.com`)

Self-hosting Langfuse is optional and not required for this slice.

## Steps

1. Ensure policy clauses are ingested (see README).
2. Submit a claim via UI (`http://localhost:8000/ui/`) or:

```bash
curl -X POST http://localhost:8000/claims \
  -F "policy_pdf=@fixtures/sample_policy.pdf" \
  -F "estimate_pdf=@fixtures/sample_estimate.pdf" \
  -F "narrative=Front-end collision damaged the bumper and headlight."
```

3. Poll until `status=completed`; note `claim_id`.
4. In the Langfuse project UI, find the trace/session with that `claim_id`.
5. Confirm child spans: `document`, `vision`, `verifiers`, `rag`, `fraud_risk`,
   `adjudicator`, and nested generation `adjudicator_llm` with token usage.

## Negative check

With Langfuse keys **unset**, the same claim still completes; no tracing errors
in worker logs that fail the claim.
