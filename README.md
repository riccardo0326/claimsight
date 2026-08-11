# ClaimSight

Multi-agent insurance claims triage. This repository currently ships **Slices 1–4**:
ingestion via FastAPI + Celery, Document Agent field extraction, a RAG Agent that
retrieves policy clauses from Postgres + pgvector (hard-filtered by `policy_id`),
a Vision Agent for optional damage photos, External Verifiers (NHTSA + Nominatim +
NWS), and a Fraud/Risk Agent (zero-shot signal + deterministic cross-checks).

Later slices (Adjudicator, LangGraph parallel branching, Langfuse, UI) are
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
                              GET /claims/{id}  ◄── completed + document_agent
                                                     + vision + verifiers
                                                     + rag + risk
```

Pipeline remains **sequential** Celery (LangGraph parallel branching is deferred).

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

## Local development / tests

```bash
python3.12 -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"

python fixtures/generate_fixtures.py
python fixtures/generate_image_fixtures.py

# Default suite: offline (no real NHTSA/Nominatim/NWS; no HF downloads)
pytest

# Optional real Hugging Face smoke tests
pytest -m hf

# Optional real external API tests (network)
pytest -m live_api
# or: python scripts/verify_verifiers_live.py
```

## Configuration

See [`.env.example`](.env.example). Slice 4 additions:

- `FRAUD_ZERO_SHOT_MODEL` — default `typeform/distilbert-base-uncased-mnli`
- `HTTP_USER_AGENT` — required by Nominatim / NWS
- `EXTERNAL_API_TIMEOUT_SECONDS` / `EXTERNAL_API_MAX_ATTEMPTS`
- `NHTSA_CACHE_TTL_HOURS` — VIN/recalls/complaints/geocode TTL (weather has none)
- `WEATHER_STORM_PRECIP_MM` — storm heuristic threshold

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)
- [docs/DECISIONS.md](docs/DECISIONS.md)
- [docs/VERIFIERS_LIVE_VERIFY.md](docs/VERIFIERS_LIVE_VERIFY.md) — live NHTSA/Nominatim/NWS checks
- [fixtures/images/README.md](fixtures/images/README.md) — manual Vision verification
