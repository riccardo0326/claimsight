# ClaimSight

Multi-agent insurance claims triage. This repository currently ships **Slices 1–3**:
ingestion via FastAPI + Celery, Document Agent field extraction, a RAG Agent that
retrieves policy clauses from Postgres + pgvector (hard-filtered by `policy_id`),
and a Vision Agent that extracts damage signal from optional claim photos using
zero-shot Hugging Face models.

Later slices (Fraud/Risk, Adjudicator, LangGraph parallel branching, Langfuse, UI)
are intentionally out of scope here.

## Architecture (this slice)

```
POST /claims (policy.pdf + estimate.pdf + narrative + optional damage_photos)
        │
        ▼
   FastAPI ──► Postgres (status=pending) ──► Celery/Redis
                                                    │
                                                    ▼
                                            Document Agent
                                            ├─ LayoutLM DocVQA (policy fields)
                                            └─ pdfplumber tables (line items)
                                                    │
                                                    ▼
                                              Vision Agent  (skipped if no photos)
                                            ├─ OWL-ViT zero-shot detection
                                            ├─ CLIP severity classification
                                            └─ BLIP VQA (fixed yes/no questions)
                                                    │
                                                    ▼
                                              RAG Agent
                                            └─ MiniLM embed + pgvector
                                               WHERE policy_id = ?
                                                    │
                                                    ▼
                              GET /claims/{id}  ◄── completed + document_agent
                                                     + vision + rag
```

Vision runs **sequentially after** Document Agent for now. True LangGraph parallel
Vision∥Document branching is a later slice.

## Prerequisites

- Docker + Docker Compose (required for the full stack and for RAG/pgvector tests)
- Python **3.12** for local development (host Python 3.14 is not recommended —
  the Hugging Face / PyTorch stack is more reliable on 3.12)

## Quick start (Docker)

If upgrading from an earlier slice, recreate the Postgres volume once so the
pgvector image and new columns apply:

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

(`other_policy_clauses.json` exists to prove retrieval never leaks across policies.)

### Submit a sample claim (with narrative)

```bash
curl -X POST http://localhost:8000/claims \
  -F "policy_pdf=@fixtures/sample_policy.pdf" \
  -F "estimate_pdf=@fixtures/sample_estimate.pdf" \
  -F "narrative=Front-end collision damaged the bumper and headlight; please review collision coverage."
```

### Submit with damage photos

Images are optional. Repeat `-F "damage_photos=@..."` for each file (`.jpg` / `.png`):

```bash
curl -X POST http://localhost:8000/claims \
  -F "policy_pdf=@fixtures/sample_policy.pdf" \
  -F "estimate_pdf=@fixtures/sample_estimate.pdf" \
  -F "narrative=Front-end collision; please review damage photos." \
  -F "damage_photos=@fixtures/images/synthetic_1.jpg" \
  -F "damage_photos=@fixtures/images/synthetic_2.jpg" \
  -F "damage_photos=@fixtures/images/synthetic_3.jpg"
```

Example response:

```json
{"claim_id":"…","status":"pending"}
```

A claim with **no** photos still completes; `result.vision` is `null`.

**Synthetic fixtures are structural proof only** — they will not produce meaningful
zero-shot detections. For a live sanity check with real confidences, drop 2–3
real car-damage photos into `fixtures/images/real/` and follow
[fixtures/images/README.md](fixtures/images/README.md).

### Poll for the result

```bash
curl http://localhost:8000/claims/<claim_id>
```

When processing finishes, `status` is `completed` and `result` contains:

- `document_agent` — `DocumentOutput` fields
- `extraction_meta` — DocVQA confidences / misses
- `vision` — `VisionOutput` (`detections`, `severity_tier`, `vqa_answers`,
  `low_confidence`), or `null` when no photos were uploaded
- `rag.retrieved_clauses` — `[{clause_id, text, similarity_score}, …]` scoped to the claim's `policy_id`

## Local development / tests

```bash
python3.12 -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[dev]"

# Generate / refresh synthetic fixture PDFs and images (already committed)
python fixtures/generate_fixtures.py
python fixtures/generate_image_fixtures.py

# Full suite: Slice 1 tests use SQLite; RAG tests spin up pgvector via testcontainers
# (Docker required, no GPU). Vision HF tests are deselected by default.
pytest
```

Optional real Hugging Face smoke tests (DocVQA + Vision models; downloads on first run):

```bash
pytest -m hf
```

## Sample fixtures

`fixtures/sample_policy.pdf` and `fixtures/sample_estimate.pdf` are **synthetic**
documents generated by `fixtures/generate_fixtures.py` (reportlab). Clause JSON
under `fixtures/*_policy_clauses.json` is also synthetic. Damage photos under
`fixtures/images/synthetic_*.jpg` are PIL silhouettes for pipeline smoke tests
only — see [fixtures/images/README.md](fixtures/images/README.md). No real customer PII.

| Field | Value |
|-------|-------|
| policy_id | `POL-2024-0098213` |
| coverage_limits | collision 50000 / comprehensive 25000 / liability 100000 |
| deductible | 1000.0 |
| vin | `1HGCM82633A004352` |
| incident_date | 2024-03-14 |
| line_items | 4 rows (bumper, headlight, labor, paint) |

## Configuration

See [`.env.example`](.env.example). Important variables:

- `DB_URL` — SQLAlchemy URL (Postgres+pgvector in Docker, SQLite for Slice 1 unit tests)
- `REDIS_URL` — Celery broker + result backend
- `STORAGE_DIR` — shared volume for uploaded PDFs and damage photos
- `DOC_QA_MODEL` — default `impira/layoutlm-document-qa`
- `DOC_QA_MIN_CONFIDENCE` — answers below this score become `None` (default `0.5`)
- `EMBEDDING_MODEL` — default `sentence-transformers/all-MiniLM-L6-v2`
- `RAG_TOP_K` — number of clauses returned (default `5`)
- `VISION_DETECTION_MODEL` — default `google/owlvit-base-patch32`
- `VISION_CLASSIFICATION_MODEL` — default `openai/clip-vit-base-patch32`
- `VISION_VQA_MODEL` — default `Salesforce/blip-vqa-base`
- `VISION_DETECTION_THRESHOLD` — detection score floor (default `0.15`)
- `VISION_LOW_CONFIDENCE_THRESHOLD` — sets `low_confidence` on severity (default `0.4`)
- `MAX_UPLOAD_MB` — upload size limit per file (default `10`)

## Document + RAG + Vision notes

- Policy fields use Hugging Face `document-question-answering` with
  `word_boxes` from pdfplumber (no Tesseract OCR required).
- Estimate `line_items` use pdfplumber table extraction — a **placeholder** for
  a proper Table Question Answering model in a later task.
- Clause embeddings use LlamaIndex `HuggingFaceEmbedding` (MiniLM); retrieval is
  SQLAlchemy + pgvector with a hard `WHERE policy_id = ?` filter.
- `retrieved_precedents` from the full §6.3 contract is deferred (see
  [docs/DECISIONS.md](docs/DECISIONS.md)).
- Vision uses zero-shot OWL-ViT / CLIP / BLIP (no fine-tuned car-damage model yet;
  see D14 in [docs/DECISIONS.md](docs/DECISIONS.md)). Severity across photos uses
  max-severity-wins.
- Schema creation uses `Base.metadata.create_all()`; Alembic migrations are
  deferred.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)
- [docs/DECISIONS.md](docs/DECISIONS.md)
- [fixtures/images/README.md](fixtures/images/README.md) — manual Vision verification
