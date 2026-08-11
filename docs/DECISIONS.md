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
