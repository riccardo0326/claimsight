# ClaimSight — Design Decisions

Log of deliberate deviations and locked tradeoffs. Future slices must read this
alongside `PROJECT_SPEC.md` and `ARCHITECTURE.md` — do not "fix" these back
to an earlier wording of the contract without an explicit decision here.

---

## Slice 1 — Ingestion + Document Agent (2026-08)

### D1. `DocumentOutput.policy_id` and `deductible` are Optional

**Spec was:** required `str` / `float`.
**Chose:** `str | None` / `float | None` (same as `vin` / `incident_date`).

Low-confidence or unparseable DocVQA answers must become `None` rather than a
hallucinated value. Making those fields required would force either failing the
whole claim or inventing a placeholder. Optional fields keep the agent honest
and leave routing-to-review for a later Adjudicator/guardrail slice.

### D2. Estimate `line_items` via pdfplumber, not Table Question Answering

**Spec listed:** HF Table Question Answering for the Document Agent.
**Chose:** `pdfplumber.extract_tables()` for this slice, with an explicit code
comment that it is a placeholder.

Keeps Slice 1 reviewable and offline-testable. A real TableQA model is a later
task; do not replace pdfplumber mid-slice without updating this decision.

### D3. `coverage_limits` shape is `{coverage_name: float}`

**Spec said:** `dict` with no key/value contract.
**Chose:** keys `collision` | `comprehensive` | `liability`, values `float`.
Keys that miss the confidence threshold are omitted (not set to `None` inside
the dict).

Gives RAG / Adjudicator a stable shape without inventing a nested schema.

### D4. Extraction confidence lives outside `DocumentOutput`

**Chose:** claim `result` JSON is:

```json
{
  "document_agent": { /* DocumentOutput */ },
  "extraction_meta": {
    "confidences": { "policy_id": 0.99, "...": "..." },
    "low_confidence_fields": [],
    "min_confidence": 0.5
  }
}
```

Keeps §6.2's output model clean for downstream agents while still persisting
scores for debugging and eval. Do not fold confidence into `DocumentOutput`
unless a later slice explicitly needs it in `ClaimState`.

### D5. Failed claims store the error in `result`, not a dedicated column

**Spec'd `claims` columns:** `id`, `status`, `created_at`, `updated_at`,
`input_paths`, `result` — no `error` field.
**Chose:** on `status=failed`, write `result = {"error": "<message>"}`.

Avoids a schema change for Slice 1. Revisit (add `error` text column) if
operators need to query failures without parsing JSONB.

### D6. DocVQA via `word_boxes`, no Tesseract / page rasterization

**Chose:** feed LayoutLM (`impira/layoutlm-document-qa`) with pdfplumber word
boxes normalized to 0..1000; pass `image=None`.

Fixture PDFs are digital-born, so the text layer is exact. Skipping OCR removes
Tesseract/poppler from the stack (painful on Windows, unnecessary for this
slice). Scanned-PDF OCR can be added later if real claim scans enter the golden
set.

### D7. Tests use a fake DocVQA extractor by default

**Chose:** `DocVQAExtractor` protocol + `FakeDocVQAExtractor` for default
`pytest`; real model behind `pytest -m hf`. Integration tests use SQLite +
Celery eager mode (no Docker required for `pytest`).

Exact-match field assertions are deterministic; the HF path remains available
as a loose smoke test without flaking CI.

### D8. Schema via `create_all`, Alembic deferred

**Chose:** `Base.metadata.create_all()` on API/worker startup (plus
`CREATE EXTENSION vector` on Postgres). Alembic migrations start when a later
slice needs non-trivial schema evolution beyond create_all-friendly adds.
Recreate the Compose Postgres volume when switching images or adding columns
(`docker compose down -v`).

---

## Slice 2 — pgvector Ingestion + RAG Agent (2026-08)

### D9. `retrieved_precedents` deferred

**Spec §6.3 included:** `retrieved_precedents: [...]` on `RAGOutput`.
**Chose:** omit entirely for this slice — no stub / empty list field.

Precedent retrieval needs a separate corpus and eval story. Shipping a fake
list would train callers to ignore grounding. Add the field when precedents
are real.

### D10. LlamaIndex for embed; SQLAlchemy + pgvector for retrieve

**Chose:** LlamaIndex `Document` + `HuggingFaceEmbedding` at ingest/query time;
persist and query via SQLAlchemy/`pgvector` with
`WHERE policy_id = :id ORDER BY embedding.cosine_distance(...)`.

Keeps the hard policy filter undeniable in SQL (critical for the cross-policy
leak test) without fighting LlamaIndex metadata-filter semantics.

### D11. RAG tests use testcontainers pgvector

**Chose:** Slice 1 Document Agent tests stay on SQLite + fake DocVQA + stubbed
RAG. Ingest / RAG / leak / narrative integration tests start
`pgvector/pgvector:pg16` via testcontainers. Full `pytest` needs Docker; no GPU.

### D12. Optional empty `narrative` on POST /claims

**Chose:** `narrative: str = Form("")` stored on `claims.narrative`.

Real claims should require a narrative (code comment). Empty is allowed so
Slice 1 curl flows keep working; RAG still synthesizes a query from
`extracted_fields` when narrative is blank.

### D13. Clause corpus is fixture JSON, not PDF chunking yet

**Chose:** ingest `fixtures/*_policy_clauses.json` rather than chunking policy
PDFs with LlamaIndex splitters.

Proves per-policy retrieval end-to-end. PDF→clause chunking arrives when the
golden corpus expands beyond hand-authored clauses.

---

## Slice 3 — Vision Agent (2026-08)

### D14. Zero-shot Vision models; no bbox; null when no photos

**Spec / architecture listed:** Object Detection + Image Classification (implying
fine-tuned or task-specific car-damage models), and
`VisionOutput.detections: [{label, bbox, confidence}]`.

**Chose for this slice:**

1. **Zero-shot models** — no labeled car-damage dataset or fine-tuning is in
   scope yet:
   - Zero-shot object detection: `google/owlvit-base-patch32` with OWL-ViT
     prompt templating (`a photo of a/an {label}`) and scene/part labels
     `["damaged car", "crashed car", "shattered windshield", "broken headlight",
     "deployed airbag", "broken glass"]`. Fine-grained attributes (`dent`,
     `scratch`, `bumper damage`) were dropped after real-photo calibration:
     raw scores stayed ≪ 0.1 while templated scene labels cleared a raised floor.
   - Zero-shot image classification: `openai/clip-vit-base-patch32` for
     `severity_tier` with `["minor damage", "moderate damage", "severe damage"]`
   - VQA: `Salesforce/blip-vqa-base` via `BlipProcessor` +
     `BlipForQuestionAnswering` (not `pipeline("visual-question-answering")`,
     which was removed in transformers 5.x) for fixed yes/no diagnostic questions
2. **Detections omit `bbox`** — contract uses
   `{label, confidence, image_path}` so output is Adjudicator-ready without
   requiring box post-processing. Add bbox when a fine-tuned detector lands.
3. **Severity aggregation: max-severity-wins** — across photos, take the highest
   tier (`minor < moderate < severe`); `severity_confidence` is that image's CLIP
   score. Conservative default for triage.
4. **`result.vision = null`** when no damage photos are uploaded (skip the
   agent entirely). Empty detections list is valid when photos exist but scores
   stay below the detection floor (`VISION_DETECTION_THRESHOLD`, default 0.45 —
   set above undamaged false-positive peaks ~0.43 for `damaged car` when that
   label is queried without a competing plain `car` anchor).
5. **Known miss:** some real views (e.g. tight side-panel damage) still produce
   no OWL-ViT hits above the floor; severity/VQA remain the fallback signal.
   Regression guards for the floor, label list, and prompt templating live in
   `tests/test_vision_agent.py` (default suite, mocked OWL-ViT — not `hf`).

Do not swap in fine-tuned detectors mid-slice without updating this decision.
True LangGraph parallel Vision∥Document branching is a later slice; the Celery
task runs Vision sequentially after Document for now.

---

## Slice 4 — External Verifiers + Fraud/Risk (2026-08)

### D15. Optional `incident_location` on POST /claims

**Spec §6.5 required:** VIN, incident date, incident location for External Verifiers.
**Chose:** add optional form field `incident_location: str | None`, persisted on
`claims.incident_location`. Do not infer location from PDFs or narrative in this
slice.

Missing location does **not** fail the claim: geocoding and weather are skipped
and `weather_at_incident = None` (no `geocoding`/`weather` entries in
`sources_failed` for a deliberate skip).

### D16. NHTSA VIN decode chains to recalls/complaints

Decode VIN via vPIC `DecodeVinValues` first. Recalls and complaints are queried
only with resolved `make` / `model` / `model_year` from that decode — never with
raw VIN alone. If decode fails or yields incomplete identity, skip
recalls/complaints and record `nhtsa_vin` in `sources_failed`.

### D17. Weather via Nominatim + NWS station observations

1. Nominatim geocodes `incident_location` → lat/lon (descriptive User-Agent;
   1 req/sec; 5s timeout).
2. NOAA/NWS `api.weather.gov/points/{lat},{lon}` → nearest observation stations.
3. `stations/{id}/observations?start=&end=` for the `incident_date` UTC day.

**Field mapping:**
- `condition` ← first non-empty `properties.textDescription`
- `precipitation_mm` ← max of `precipitationLastHour|3Hours|6Hours` converted to
  mm (NWS QuantitativeValues in meters → ×1000; inches → ×25.4)
- `had_storm_event` ← True if precip ≥ `WEATHER_STORM_PRECIP_MM` (default 5.0) OR
  text/presentWeather mentions thunder/storm/hail/tornado/severe

If location or `incident_date` is missing, weather is skipped. API failures set
`weather_at_incident = None` and add `weather` (or `geocoding`) to
`sources_failed`. Never invent weather.

### D18. `external_api_cache` (SQLite + Postgres)

Table: `cache_key` (unique), `source`, `response_json` (PortableJSON),
`fetched_at`, `expires_at` (nullable).

| Source | Cache key | TTL |
|---|---|---|
| NHTSA VIN | `nhtsa_vin:{VIN}` | 24h (`NHTSA_CACHE_TTL_HOURS`) |
| Recalls | `nhtsa_recalls:{make}:{model}:{year}` | 24h |
| Complaints | `nhtsa_complaints:{make}:{model}:{year}` | 24h |
| Geocoding | `geocoding:{normalized location}` | 24h |
| Weather | `weather:{lat:.4f}:{lon:.4f}:{date}` | **none** (`expires_at=NULL`) |

Only successful HTTP responses are cached. Created on SQLite and Postgres via
`create_all` / explicit SQLite create (same pattern as `claims`).
Recall/complaint lists are capped at 50 each when mapping into `VerifierOutput`
so a single claim result stays reviewable.

### D19. HTTP resilience

All outbound calls: timeout 5s, max 2 attempts, Tenacity exponential backoff.
Source-level failures populate `sources_failed` with stable ids
(`nhtsa_vin`, `nhtsa_recalls`, `nhtsa_complaints`, `geocoding`, `weather`) and
**never** fail the claim. Exception traces stay in logs, not in Pydantic output.

### D20. Fraud/Risk: zero-shot model + heuristic score

- Model: `typeform/distilbert-base-uncased-mnli` (`FRAUD_ZERO_SHOT_MODEL`) —
  lightweight MNLI zero-shot compatible with the existing Transformers stack.
- Candidate labels: consistent / inconsistent / possible staged damage /
  weather mismatch / recall-related damage. Classifier is a **signal only**.
- Deterministic rules: weather mismatch (narrative weather cause +
  `had_storm_event is False` → severity `medium`); recall-related damage
  (recall component token overlap with narrative/line items → severity `info`).
- Risk score: weighted sum clamped to `[0, 1]` — staged 0.40, inconsistent 0.35,
  weather rule 0.30, recall rule 0.10; classifier-only weather/recall use lower
  weights. Matching rule + classifier evidence is not double-counted.
  **Not statistically calibrated** — portfolio heuristic.

Empty narrative → no classifier flags; rules may still fire.

### D21. LangGraph remains deferred

Architecture still targets a LangGraph orchestrator with parallel Vision ∥
Document ∥ Verifiers. Slice 4 keeps the Celery task sequential:

`Document → Vision → Verifiers → RAG → Fraud/Risk → persist`

(Slice 5 appends Adjudicator — see D29.)

Fraud/Risk does not depend on RAG. True parallel branching stays a later slice.

### D22. Result keys `verifiers` + `risk`

`claim.result` gains `verifiers` (`VerifierOutput`) and `risk` (`RiskOutput`)
alongside existing `document_agent`, `extraction_meta`, `vision`, `rag`. Nested
shapes follow the Slice 4 task contracts (richer than the §6.5 ellipsis stubs).

---

## Slice 5 — Adjudicator + Citation Guardrails (2026-08)

### D23. Frontier provider: OpenAI Chat Completions via httpx

**Chose:** OpenAI-compatible Chat Completions (`ADJUDICATOR_BASE_URL`, default
`https://api.openai.com/v1`) using existing `httpx` + Tenacity — no official
OpenAI SDK. Env: `OPENAI_API_KEY`, `ADJUDICATOR_MODEL` (default `gpt-4o`),
`ADJUDICATOR_TIMEOUT_SECONDS` (default 60).

Matches the locked “GPT-4-class for Adjudicator” stack without expanding deps.

### D24. Structured output: JSON + Pydantic; no silent repair

The model is prompted for JSON-only `ClaimReport`. We parse and
`ClaimReport.model_validate`. Malformed JSON / invalid enum / missing fields →
deterministic `needs_review` fallback (`fallback_needs_review`). Do not invent
approve/deny from broken output.

### D25. Persist key `adjudication`; status stays `completed`

`claim.result.adjudication` holds the `ClaimReport` JSON
(`model_dump(mode="json")`). Human review is expressed as
`decision=needs_review` inside that object.

**Architecture** mentioned claim status `needs_human_review`. **Slice 5 does
not** add that enum value — claim `status` remains `completed` when the
pipeline finishes (including review decisions). Polling clients read
`result.adjudication.decision`.

### D26. `risk_flags` copied from upstream Fraud/Risk

`ClaimReport.risk_flags` is always overwritten with `RiskOutput.flags` after
the LLM returns. The Adjudicator does not re-run Fraud/Risk or invent flags.
Risk score / flags are signals — never automatic proof of fraud / auto-deny.

### D27. Deterministic confidence heuristic (not calibrated)

Final `confidence` is recomputed by `compute_confidence` after guardrails:

- start `0.55`
- `+0.15` if RAG non-empty, citations valid, and ≥1 citation present
- `+0.05` if RAG usable but decision is not approve and cites may be empty
- `+0.10` if vision present and (detections non-empty OR not `low_confidence`)
- `-0.15` if critical low-confidence extraction (`policy_id` / `coverage_limits.*`)
- `-0.10` if any `sources_failed`
- `-0.20` if `needs_review` / guardrail override
- `-0.10` if any medium/high risk flag
- clamp `[0, 1]`

LLM self-reported confidence is ignored for the persisted value.

### D28. Guardrail policy (approve / deny / review)

- **Citation subset (hard):** `cited_clauses ⊆ retrieved clause_ids` or force review.
- **Approve** requires non-empty RAG and ≥1 valid citation; empty RAG ≠ deny.
- **Deny** requires retrieved policy evidence (empty RAG → review, not deny).
- Empty Vision `detections` / `vision=null` never alone force deny (no-signal ≠ no damage).
- `sources_failed` is missing evidence (lowers confidence), not a negative finding.
- Material risk flags (`weather mismatch`, `possible staged damage`,
  `inconsistent claim`) block approve → `needs_review`.
- Low-confidence critical extraction blocks approve → `needs_review`.

### D29. Celery order after Slice 5

`Document → Vision → Verifiers → RAG → Fraud/Risk → Adjudicator → guardrails → persist`

LangGraph remains deferred (D21). LLM/schema/guardrail failures persist
`adjudication.decision=needs_review` and keep claim `status=completed` — they do
not fail the claim.

Prompt template: `prompts/adjudicator_v1.md`. Live verify:
`python scripts/verify_adjudicator_live.py` / `pytest -m live_llm`.

---

## Slice 6 — Golden Dataset + Eval Harness (2026-08)

Completes the unfinished half of PROJECT_SPEC milestone 4: a synthetic golden
set (~50) and a first measurable Adjudicator eval run. Langfuse, CI gates,
LangGraph, RAGAS, and UI remain deferred.

### D30. Eval surface is Adjudicator + guardrails on canned upstream

**Chose:** Score `run_adjudicator` against golden cases that embed canned
`document_agent` / `extraction_meta` / `vision` / `verifiers` / `rag` / `risk`
snapshots. Do **not** require full Celery + HF + live NHTSA for the default
eval path.

Keeps the harness fast, offline-testable, and focused on the decision/citation
contract. Full pipeline E2E remains covered by Slice 4/5 integration tests.

### D31. Golden size ~50; schema adapts §8.1 to shipped contracts

**Spec §8.1:** singular `ground_truth_clause_id`; target later grows to ≥150.
**Chose:** `fixtures/golden/manifest.jsonl` with ~50 synthetic cases;
`ground_truth.clause_ids: list[str]`; `ground_truth.decision` ∈
`approve|deny|needs_review` (D25); `ground_truth.fraud_flag: bool` = any
material risk flag in `{weather mismatch, possible staged damage,
inconsistent claim}` (same set as Adjudicator guardrails).

Regenerate via `python fixtures/golden/build_manifest.py`. Transparency about
synthetic data is intentional (PROJECT_SPEC §7).

### D32. Metrics in Slice 6; faithfulness / cost / CI deferred

**Measured now:**

- Post-guardrail citation hallucination rate (target **0%**)
- Decision accuracy (exact match; report baseline — do not block the slice on
  ≥85% live accuracy yet)
- Fraud-flag precision/recall vs `fraud_flag` (baseline only)

**Deferred:** RAGAS / LLM-judge faithfulness ≥ 0.85; cost-per-claim and P95
latency (with Langfuse); GitHub Actions §8.3 PR gate; expanding to 150–300
cases.

### D33. Default pytest stays offline; live eval is opt-in

**Chose:** `python scripts/run_eval.py --mode fake` uses an oracle stub LLM
keyed to each case’s ground truth (legal citations only) for harness smoke and
the checked-in sample report under `eval/reports/`. `--mode live` uses the
existing OpenAI client (`OPENAI_API_KEY`). Default `pytest` covers
`tests/test_eval_*.py` only — never the full live golden run.

Package: `eval/` (`schema`, `metrics`, `runner`, `report`). Docs:
`docs/EVAL.md`, `fixtures/golden/README.md`.
