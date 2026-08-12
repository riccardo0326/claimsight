# ClaimSight

Multi-agent insurance claims triage. This repository currently ships **Slices 1–6**:
ingestion via FastAPI + Celery, Document Agent field extraction, a RAG Agent that
retrieves policy clauses from Postgres + pgvector (hard-filtered by `policy_id`),
a Vision Agent for optional damage photos, External Verifiers (NHTSA + Nominatim +
NWS), a Fraud/Risk Agent (zero-shot signal + deterministic cross-checks), an
Adjudicator (frontier LLM synthesis + citation grounding guardrails), and a
**golden dataset + offline-first eval harness** for measurable Adjudicator quality.

Later slices (LangGraph parallel branching, Langfuse, CI eval gate, UI) are
intentionally out of scope here.

## Architecture (this slice)

```
POST /claims (policy.pdf + estimate.pdf + narrative
              + optional damage_photos + optional incident_location)
        │
        ▼
   FastAPI ──► Postgres (status=pending) ──► Celery/Redis
                                                    │
                                                    ▼
                                            Document Agent
                                                    │
                                                    ▼
                                              Vision Agent  (skipped if no photos)
                                                    │
                                                    ▼
                                           External Verifiers
                                            ├─ NHTSA VIN → recalls/complaints
                                            ├─ Nominatim geocode (if location)
                                            └─ NWS observations (if location+date)
                                                    │
                                                    ▼
                                              RAG Agent
                                                    │
                                                    ▼
                                           Fraud/Risk Agent
                                            ├─ zero-shot narrative labels
                                            └─ weather / recall rules
                                                    │
                                                    ▼
                                            Adjudicator
                                            ├─ frontier LLM (OpenAI) proposes ClaimReport
                                            └─ deterministic citation/schema guardrails
                                                    │
                                                    ▼
                              GET /claims/{id}  ◄── completed + document_agent
                                                     + vision + verifiers
                                                     + rag + risk + adjudication

Offline eval (Slice 6):
  fixtures/golden/manifest.jsonl ──► eval runner ──► Adjudicator+guardrails
                                                 ──► eval/reports/latest.{json,md}
```

Pipeline remains **sequential** Celery (LangGraph parallel branching is deferred).
Human review is `result.adjudication.decision = needs_review` (claim `status`
stays `completed`). Eval scores Adjudicator+guardrails on **canned upstream**
snapshots (not a full Celery/HF re-run).

## Prerequisites

- Docker + Docker Compose (required for the full stack and for RAG/pgvector tests)
- Python **3.12** for local development (host Python 3.14 is not recommended —
  the Hugging Face / PyTorch stack is more reliable on 3.12)

## Quick start (Docker)

If upgrading from an earlier slice, recreate the Postgres volume once so new
columns/tables apply:

```bash
docker compose down -v
cp .env.example .env
docker compose up --build
```

Services:

| Service   | URL / port        |
|-----------|-------------------|
| API       | http://localhost:8000 |
| Postgres  | localhost:5432 (pgvector/pgvector:pg16) |
| Redis     | localhost:6379    |
| Worker    | (background)      |

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

## Golden eval (Slice 6)

```bash
# Deterministic oracle LLM — no network; writes eval/reports/latest.*
python scripts/run_eval.py --mode fake

# Live frontier LLM (requires OPENAI_API_KEY)
python scripts/run_eval.py --mode live
```

Metrics: post-guardrail citation hallucination rate, decision accuracy, fraud-flag
precision/recall. See [docs/EVAL.md](docs/EVAL.md) and
[fixtures/golden/README.md](fixtures/golden/README.md).

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

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)
- [docs/DECISIONS.md](docs/DECISIONS.md)
- [docs/EVAL.md](docs/EVAL.md) — Slice 6 golden eval harness
- [docs/VERIFIERS_LIVE_VERIFY.md](docs/VERIFIERS_LIVE_VERIFY.md) — live NHTSA/Nominatim/NWS checks
- [docs/ADJUDICATOR_LIVE_VERIFY.md](docs/ADJUDICATOR_LIVE_VERIFY.md) — live OpenAI Adjudicator checks
- [fixtures/images/README.md](fixtures/images/README.md) — manual Vision verification
- [fixtures/golden/README.md](fixtures/golden/README.md) — golden dataset schema
