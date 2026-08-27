# ClaimSight

Multi-agent insurance claims triage: ingest policy + estimate PDFs, an optional
narrative / location / damage photos, run a LangGraph agent pipeline, and return
a citation-grounded `approve` / `deny` / `needs_review` recommendation.

This README reflects the **current repo** against
[`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) (contracts, eval, Definition of
Done). Design tradeoffs live in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Implementation status

**Shipped (Slices 1–9 + 11):** end-to-end API + worker, all agent nodes from
spec §6 (with documented deferrals), LangGraph orchestration, golden eval + CI
gate, optional Langfuse, Docker Compose, and a static demo UI.

| Area | Status | Notes |
|------|--------|-------|
| FastAPI ingest + Celery worker + Postgres/Redis | Done | `POST /claims`, `GET /claims/{id}` |
| Document Agent (LayoutLM DocVQA) | Done | Estimate `line_items` via pdfplumber (TableQA deferred) |
| RAG Agent (LlamaIndex embed + pgvector) | Done | Hard `policy_id` filter; `retrieved_precedents` deferred |
| Vision Agent (OWL-ViT / CLIP / BLIP) | Done | Zero-shot; no bboxes; `null` when no photos |
| External Verifiers (NHTSA + Nominatim + NWS) | Done | Degrades into `sources_failed`; optional location |
| Fraud/Risk Agent (zero-shot + rules) | Done | |
| Adjudicator + citation guardrails | Done | Needs `OPENAI_API_KEY` for live runs; else falls back to `needs_review` |
| LangGraph orchestrator | Done | Vision ∥ Document; Verifiers/RAG after Document; Adjudicator join |
| Golden eval + GitHub Actions gate | Done | **50** synthetic cases; offline `--mode fake` gate |
| Langfuse tracing | Done | Optional; claim_id-rooted spans + Adjudicator token usage |
| Demo UI | Done | Static `/ui/` (not Next.js/Streamlit — see D47) |
| Docker Compose packaging | Done | |

### Latest offline eval (`--mode fake`, n=50)

| Metric | Value |
|--------|------:|
| Decision accuracy | 1.000 |
| Citation hallucination rate (post-guardrail) | 0.000 |
| Fraud-flag precision / recall | 1.000 / 1.000 |

Source: [`eval/reports/latest.md`](eval/reports/latest.md). Fake mode uses an
oracle stub — accuracy gates harness/GT consistency, not live model quality
(see [`docs/EVAL.md`](docs/EVAL.md)).

## What’s missing for a fully portfolio-ready app

Mapped to [PROJECT_SPEC §10 Definition of Done](docs/PROJECT_SPEC.md) and
deferred items in `DECISIONS.md`:

| Gap | Spec expectation | Current state |
|-----|------------------|---------------|
| Golden set size | ≥150 labeled claims | **50** cases in `fixtures/golden/manifest.jsonl` |
| Faithfulness metric | RAGAS or LLM-judge ≥ 0.85; CI fails if regresses > 0.05 | **Not implemented** — not measured or gated |
| Cost / latency metrics | Tracked cost-per-claim + P95 under 30s in eval/report | Routing narrative in `docs/COST_ROUTING.md`; **no checked-in live numbers or P95 gate** |
| Demo media | README demo GIF/video | **Missing** (optional per D47) |
| TableQA for estimates | HF Table Question Answering | **pdfplumber** tables only (D2) |
| Precedent retrieval | `retrieved_precedents` on RAG | **Deferred** (D9) |
| Vision detections | `{label, bbox, confidence}` + optional fine-tune | **Zero-shot labels/confidence only; no bbox** (D14) |
| Live Adjudicator in CI | Spec CI mentions live quality signals | CI is **offline fake only**; `pytest -m live_llm` / `--mode live` are manual |
| Full-pipeline golden scoring | Eval over real Celery/HF E2E | Eval scores **Adjudicator + guardrails on canned upstream** only |
| Schema migrations | Durable schema evolution | `create_all` + ad-hoc alters; **Alembic deferred** (D8) |
| Clause corpus from PDFs | Chunk real policy PDFs | Fixture JSON ingest only (D13) |
| Police report upload | Mentioned in Architecture ingest list | **Not in API** — policy + estimate + photos + narrative only |
| Frontend stack | Spec: Next.js or Streamlit | **Static FastAPI-mounted UI** by design (D47) |
| K8s | Optional stretch | **Not present** — Compose only |

**Already satisfied for demo use:** API + UI submit → structured report; all
five agent roles + verifiers; CI hallucination/accuracy gates; optional
Langfuse per `claim_id`; `DECISIONS.md` tradeoff log.

**Required for a live end-to-end claim (not just offline eval):** Docker stack
up, clause fixtures ingested, Hugging Face models downloadable on the worker,
and `OPENAI_API_KEY` set for a real Adjudicator decision (without it the
guardrail path returns `needs_review`).

## Architecture (current)

```
POST /claims (policy.pdf + estimate.pdf + narrative
              + optional damage_photos + optional incident_location)
        │
        ▼
   FastAPI ──► Postgres (status=pending) ──► Celery/Redis
                                                    │
                                                    ▼
                                            LangGraph
                         ┌──────── Document Agent ────────┐
                         │         ├─ RAG Agent           │
                         │         └─ External Verifiers  │
                         │                 └─ Fraud/Risk  │
                         └──────── Vision Agent ──────────┘
                                          │
                                          ▼
                                    Adjudicator
                         ├─ frontier LLM (OpenAI) proposes ClaimReport
                         └─ deterministic citation/schema guardrails
                                          │
                                          ▼
                              GET /claims/{id}  +  Demo UI /ui/
                                   completed + document_agent
                                   + vision + verifiers + rag
                                   + risk + adjudication

Offline eval (Slices 6–7):
  fixtures/golden/manifest.jsonl ──► eval runner ──► Adjudicator+guardrails
                                                 ──► eval/reports/latest.{json,md}
                                                 ──► --gate vs baseline_fake.json (CI)

Observability (Slice 8, optional Langfuse):
  process_claim ──► spans (document/vision/verifiers/rag/fraud_risk/adjudicator)
                 ──► generation adjudicator_llm (+ token usage)
```

Pipeline is **LangGraph** inside one Celery job (D40). Human review is
`result.adjudication.decision = needs_review` (claim `status` stays
`completed`). Eval scores Adjudicator+guardrails on **canned upstream**
snapshots (not a full Celery/HF re-run).

## Prerequisites

- Docker + Docker Compose (required for the full stack and for RAG/pgvector tests)
- Python **3.12** for local development (host Python 3.14 is not recommended —
  the Hugging Face / PyTorch stack is more reliable on 3.12)
- `OPENAI_API_KEY` for live Adjudicator decisions (optional Langfuse keys for traces)

## Quick start (Docker)

If upgrading from an earlier slice, recreate the Postgres volume once so new
columns/tables apply:

```bash
docker compose down -v
cp .env.example .env
# Edit .env: set OPENAI_API_KEY for live adjudication
docker compose up --build
```

Services:

| Service   | URL / port        |
|-----------|-------------------|
| API + UI  | http://localhost:8000 (`/ui/`, `/docs`) |
| Postgres  | localhost:5432 (pgvector/pgvector:pg16) |
| Redis     | localhost:6379    |
| Worker    | (background)      |

Demo UI: http://localhost:8000/ui/  
OpenAPI docs: http://localhost:8000/docs

### Ingest policy clause fixtures

After the stack is up, load both clause corpora into `policy_clauses`:

```bash
docker compose exec api python -m rag.ingest fixtures/sample_policy_clauses.json
docker compose exec api python -m rag.ingest fixtures/other_policy_clauses.json
```

### Submit a sample claim (with narrative + location)

```bash
curl -X POST http://localhost:8000/claims \
  -F "policy_pdf=@fixtures/sample_policy.pdf" \
  -F "estimate_pdf=@fixtures/sample_estimate.pdf" \
  -F "narrative=Front-end collision damaged the bumper and headlight; please review collision coverage." \
  -F "incident_location=Washington, DC"
```

`incident_location` is optional. If omitted, geocoding/weather are skipped and
`result.verifiers.weather_at_incident` is `null`.

### Poll for the result

```bash
curl http://localhost:8000/claims/<claim_id>
```

When processing finishes, `status` is `completed` and `result` contains:

- `document_agent` — `DocumentOutput` fields
- `extraction_meta` — DocVQA confidences / misses
- `vision` — `VisionOutput` or `null` when no photos
- `verifiers` — `VerifierOutput` (NHTSA + optional weather; `sources_failed` on degrade)
- `rag.retrieved_clauses` — clauses scoped to the claim's `policy_id`
- `risk` — `RiskOutput` (`flags`, `risk_score` in `[0, 1]`)
- `adjudication` — `ClaimReport` (`decision`, `confidence`, `cited_clauses`,
  `risk_flags`, `reasoning_summary`). Human review is `decision=needs_review`.

The demo UI shows the same decision, reasoning, cited clause text, and
collapsible per-agent panels (raw JSON collapsed by default).

## Golden eval + CI gate (Slices 6–7)

```bash
# Deterministic oracle LLM — no network; writes eval/reports/latest.*
python scripts/run_eval.py --mode fake

# Same + fail if hallucination > 0% or accuracy drops > 2pp vs baseline
python scripts/run_eval.py --mode fake --gate

# Live frontier LLM (requires OPENAI_API_KEY) — not used in CI
python scripts/run_eval.py --mode live
```

GitHub Actions (`.github/workflows/ci.yml`) runs default `pytest` and the fake
eval gate on every PR and on pushes to `main`. Metrics and baseline policy:
[docs/EVAL.md](docs/EVAL.md), [fixtures/golden/README.md](fixtures/golden/README.md).

## Local development / tests

```bash
python3.12 -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"

python fixtures/generate_fixtures.py
python fixtures/generate_image_fixtures.py
# optional: python fixtures/golden/build_manifest.py

# Default suite: offline (no real NHTSA/Nominatim/NWS; no HF; no OpenAI)
pytest

# Optional real Hugging Face smoke tests
pytest -m hf

# Optional real external API tests (network)
pytest -m live_api
# or: python scripts/verify_verifiers_live.py

# Optional real Adjudicator frontier LLM (requires OPENAI_API_KEY)
pytest -m live_llm
# or: python scripts/verify_adjudicator_live.py
```

## Configuration

See [`.env.example`](.env.example). Notable settings:

- `FRAUD_ZERO_SHOT_MODEL` — default `typeform/distilbert-base-uncased-mnli`
- `HTTP_USER_AGENT` — required by Nominatim / NWS
- `EXTERNAL_API_TIMEOUT_SECONDS` / `EXTERNAL_API_MAX_ATTEMPTS`
- `NHTSA_CACHE_TTL_HOURS` — VIN/recalls/complaints/geocode TTL (weather has none)
- `WEATHER_STORM_PRECIP_MM` — storm heuristic threshold
- `OPENAI_API_KEY` — required for live Adjudicator (never commit)
- `ADJUDICATOR_MODEL` — default `gpt-4o`
- `ADJUDICATOR_BASE_URL` — default `https://api.openai.com/v1`
- `ADJUDICATOR_TIMEOUT_SECONDS` — default `60`
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — optional Slice 8 tracing
- `LANGFUSE_HOST` — default `https://cloud.langfuse.com`
- `GRAPH_MAX_CONCURRENCY` — LangGraph parallel node cap (default `2`; set `1` to serialize HF)

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)
- [docs/DECISIONS.md](docs/DECISIONS.md)
- [docs/EVAL.md](docs/EVAL.md) — golden eval harness + CI gate (Slices 6–7)
- [docs/COST_ROUTING.md](docs/COST_ROUTING.md) — model routing + Langfuse cost readout
- [docs/LANGFUSE_VERIFY.md](docs/LANGFUSE_VERIFY.md) — live Langfuse trace check
- [docs/VERIFIERS_LIVE_VERIFY.md](docs/VERIFIERS_LIVE_VERIFY.md) — live NHTSA/Nominatim/NWS checks
- [docs/ADJUDICATOR_LIVE_VERIFY.md](docs/ADJUDICATOR_LIVE_VERIFY.md) — live OpenAI Adjudicator checks
- [fixtures/images/README.md](fixtures/images/README.md) — manual Vision verification
- [fixtures/golden/README.md](fixtures/golden/README.md) — golden dataset schema
