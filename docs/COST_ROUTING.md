# Cost & model routing (Slice 8)

ClaimSight already routes compute by node type. Slice 8 makes Adjudicator token
usage visible in Langfuse so portfolio reviews can cite cost-per-claim.

## De facto routing

| Node | Model class | Where it runs | Billed tokens? |
|------|-------------|---------------|----------------|
| Document Agent | LayoutLM DocVQA (HF) | Worker / local GPU-or-CPU | No (self-hosted weights) |
| Vision Agent | OWL-ViT / CLIP / BLIP (HF) | Worker | No |
| RAG embeddings | MiniLM sentence-transformers | Worker + Postgres/pgvector | No |
| Fraud/Risk zero-shot | DistilBERT MNLI (HF) | Worker | No |
| External Verifiers | NHTSA / Nominatim / NWS HTTP | Worker | No (public APIs) |
| Adjudicator | Frontier chat model (default `gpt-4o`) | OpenAI Chat Completions | **Yes** — prompt + completion tokens |

This is the “before routing” story for the portfolio: heavy multimodal /
retrieval work stays on local HF models; only synthesis uses a frontier LLM.

## Reading cost from Langfuse

1. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (and optional `LANGFUSE_HOST`).
2. Process a claim through the API/worker (or UI at `/ui/`).
3. In Langfuse, filter traces by `claim_id` / session id = claim UUID.
4. Open the `adjudicator_llm` **generation** under the `adjudicator` span.
5. Read `usage` / `usage_details` (`input` / `output` / `total` tokens) and
   Langfuse’s estimated cost (based on your project’s model price table).

Prompt template version is tagged as `prompts/adjudicator_v1.md`.

## Example metrics table (template)

Fill after one live run with your key (do not commit secrets):

| Metric | Example / method |
|--------|------------------|
| Adjudicator input tokens | from Langfuse generation `usage.input` |
| Adjudicator output tokens | from Langfuse generation `usage.output` |
| Adjudicator estimated USD | Langfuse cost for that generation |
| HF nodes token bill | $0 (self-hosted) |
| P95 end-to-end latency | claim `updated_at - created_at` or Langfuse root span duration |

**Before/after routing (portfolio narrative):** “After” is the current split
above. A pure-frontier baseline (“before”) would send Document/Vision/Fraud
classification to a paid LLM as well — expect substantially higher token cost
with little quality gain for structured extraction. We do not re-run that
baseline in CI; document the contrast qualitatively using the table.

## Related

- Live verify: [LANGFUSE_VERIFY.md](LANGFUSE_VERIFY.md)
- Eval harness still does not gate on cost (see [EVAL.md](EVAL.md))
