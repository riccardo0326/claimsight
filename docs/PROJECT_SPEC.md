# ClaimSight — Project Spec

> Reference doc for implementation. Keep this open (or linked via
> `.cursorrules`) while building in Cursor so generated code stays aligned
> with the agreed design instead of drifting toward generic boilerplate.

## 1. Problem Statement

Insurance claims triage is manual, slow, and inconsistent. A human adjuster
spends 20–40 minutes per claim cross-referencing photos, policy documents,
and estimates. ClaimSight automates first-pass triage: it ingests a claim's
raw materials and produces a structured, evidence-cited recommendation
(approve / deny / needs human review), reducing the manual review burden
for clear-cut cases while flagging ambiguous ones honestly rather than
forcing a decision.

## 2. Goals

- Ship a working end-to-end pipeline: multimodal ingestion → multi-agent
reasoning → grounded structured output.
- Every claim decision must be traceable to specific retrieved evidence
(policy clause citations) — no ungrounded claims.
- Every pipeline change must be measurable against a golden eval set before
it's considered "done."
- Demonstrate cost-aware model routing, not just "call GPT-4 for
everything."



## 3. Non-Goals

- No real customer/PII data, no compliance certification.
- No claim to be a state-of-the-art fraud detector — the fraud agent
demonstrates the *pattern* of external-data cross-referencing.
- No custom model training from scratch — fine-tuning/using pretrained HF
models is in scope; training a vision model from zero is not.



## 4. Users / Scenario

Primary "user" for demo purposes: a claims adjuster reviewing ClaimSight's
output before making a final call. Secondary "user" for portfolio purposes:
an engineering interviewer reading the README/trace/eval report.

## 5. Tech Stack (locked decisions)


| Layer             | Choice                                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| Orchestration     | LangGraph                                                                                            |
| RAG ingestion     | LlamaIndex                                                                                           |
| Vector store      | Postgres + pgvector                                                                                  |
| Vision/Doc models | Hugging Face Transformers / Inference Endpoints                                                      |
| Reasoning model   | Frontier model (Claude/GPT-4-class) for Adjudicator only; small/local model for classification nodes |
| Eval              | RAGAS + custom LLM-judge scripts                                                                     |
| Observability     | Langfuse                                                                                             |
| Backend           | FastAPI + Celery + Redis                                                                             |
| Deployment        | Docker Compose (K8s manifests optional stretch)                                                      |
| CI/CD             | GitHub Actions running eval suite on PR                                                              |
| Frontend          | Next.js or Streamlit (built last)                                                                    |


Do not substitute these without a documented reason in `DECISIONS.md` — the
point of a locked stack is to avoid mid-project churn.

## 6. Agent Graph Contract

Each node below is a hard contract: input schema, output schema, and the
model/task it uses. Cursor should implement each as an isolated,
independently testable function/module conforming to this contract.

### 6.1 Vision Agent

- **Input:** list of image file paths (damage photos)
- **Output:** `VisionOutput { detections: [{label, bbox, confidence}], severity_tier: str, vqa_answers: {question: answer} }`
- **HF tasks:** Object Detection, Image Classification, Visual Question Answering



### 6.2 Document Agent

- **Input:** policy PDF path, estimate PDF path
- **Output:**
  ```
  DocumentOutput {
    policy_id: str | None
    coverage_limits: dict[str, float]   # keys: collision | comprehensive | liability (omit misses)
    deductible: float | None
    vin: str | None
    incident_date: date | None
    line_items: list[{description: str, cost: float}]
  }
  ```
- **HF tasks:** Document Question Answering (policy fields). Table Question Answering is deferred — estimate `line_items` currently use pdfplumber table extraction (see `DECISIONS.md`).
- **Confidence rule:** DocVQA answers below `DOC_QA_MIN_CONFIDENCE` (default 0.5), plus values that fail post-parse hardening (VIN / currency / date), must be `None` / omitted — never hallucinated. Confidence scores and miss lists are stored on the claim row as `result.extraction_meta`, not inside `DocumentOutput`.



### 6.3 RAG Agent

- **Input:** `policy_id`, claim narrative, extracted fields from 6.2
- **Output:**
  ```
  RAGOutput {
    retrieved_clauses: list[{clause_id: str, text: str, similarity_score: float}]
  }
  ```
  `retrieved_precedents` from the original contract is **deferred** — see `DECISIONS.md`.
- **HF / embedding tasks:** Feature Extraction / Sentence Similarity via
  `sentence-transformers/all-MiniLM-L6-v2` (LlamaIndex `HuggingFaceEmbedding`).
- **Constraint:** must hard-filter retrieval to the claim's own `policy_id`
  (`WHERE policy_id = ?`) — never retrieve across policies.
- **Store:** Postgres + pgvector table `policy_clauses`; ingest via
  `python -m rag.ingest <clauses.json>`.



### 6.4 Fraud/Risk Agent

- **Input:** claim narrative, Document Agent output, External Verifier output
- **Output:** `RiskOutput { flags: [{flag_type, rationale, severity}], risk_score: float }`
  with `risk_score` constrained to `[0.0, 1.0]`
- **HF tasks:** Zero-Shot Classification (`typeform/distilbert-base-uncased-mnli` by
  default) plus deterministic cross-check rules (weather mismatch, recall-related).
  See `DECISIONS.md` Slice 4 for score heuristic and severity choices.



### 6.5 External Verifiers

- **Input:** VIN (from Document Agent), incident date (from Document Agent),
  optional `incident_location` (from `POST /claims` form — not inferred)
- **Output:**
  ```
  VerifierOutput {
    make: str | None
    model: str | None
    model_year: int | None
    nhtsa_recalls: list[{campaign_number, component, summary}]
    nhtsa_complaints: list[{complaint_id, component, summary}]
    weather_at_incident: {condition, precipitation_mm, had_storm_event} | None
    sources_failed: list[str]
  }
  ```
- **APIs:** NHTSA VIN Decoder (vPIC), NHTSA Complaints/Recalls (by make/model/year
  after decode), Nominatim geocoding, NOAA/NWS station observations for the
  incident date. Responses cached in `external_api_cache` (see `DECISIONS.md`).
- Not an HF task — plain REST calls via `httpx` + Tenacity; source failures degrade
  into `sources_failed` and never fail the claim.



### 6.6 Adjudicator

- **Input:** all of the above (`ClaimState` in full)
- **Output:** `ClaimReport { decision: approve|deny|needs_review, confidence: float, cited_clauses: [clause_id], risk_flags: [...], reasoning_summary: str }`
- **Hard rule:** `cited_clauses` must be a subset of `RAGOutput.retrieved_clauses` clause_ids. Violation → guardrail rejects output, claim routed to `needs_review`.
- **HF task:** Text Generation (frontier model)



## 7. Data Sources

- **Synthetic golden dataset:** 150–300 generated claims (image + PDF +
narrative) with labeled ground truth. Document generation method in
README — transparency about synthetic data is part of the credibility
story.
- **NHTSA VIN Decoder API** — public, free, no key required for basic use.
- **NHTSA Complaints/Recalls API** — public.
- **NOAA/NWS Weather API** — public, free.
- **Kaggle car damage detection dataset** — for vision model
benchmarking/fine-tuning.
- Sample policy PDF templates — publicly available insurance policy
templates, used as realistic document structure references.



## 8. Eval Harness Spec



### 8.1 Golden Dataset Schema

```
claim_id, images[], policy_pdf, estimate_pdf, narrative,
ground_truth_decision, ground_truth_clause_id, ground_truth_fraud_flag
```



### 8.2 Metrics


| Metric                      | Method                                  | Target (initial)                                                  |
| --------------------------- | --------------------------------------- | ----------------------------------------------------------------- |
| Citation hallucination rate | check cited_clauses ⊆ retrieved_clauses | 0% (hard guardrail, not just measured)                            |
| Groundedness/faithfulness   | RAGAS or LLM-judge                      | ≥ 0.85                                                            |
| Decision accuracy           | exact match vs. ground_truth_decision   | ≥ 85% on golden set                                               |
| Fraud-flag precision/recall | vs. ground_truth_fraud_flag             | report both, no fixed target initially — establish baseline first |
| Cost per claim              | sum of token costs across nodes         | track + reduce via model routing                                  |
| P95 latency per claim       | end-to-end job time                     | < 30s (excluding human review queue time)                         |




### 8.3 CI Gate

- Eval suite runs on every PR touching `agents/`, `rag/`, or `prompts/`.
- PR blocked if: hallucination rate > 0%, decision accuracy regresses > 2
points from main, or faithfulness regresses > 0.05 from main.



## 9. Milestones (suggested 1–3 month solo timeline)

1. **Week 1–2:** repo scaffold, FastAPI + Celery skeleton, Document Agent
  working end-to-end on one sample PDF.
2. **Week 3–4:** pgvector + LlamaIndex ingestion pipeline; RAG Agent
  retrieving real clauses.
3. **Week 5–6:** Vision Agent + External Verifiers; parallel branches
  wired into LangGraph.
4. **Week 7–8:** Adjudicator + guardrails; golden dataset built (even if
  small, 50 claims); first eval run.
5. **Week 9–10:** Langfuse tracing, model routing, CI eval gate.
6. **Week 11–12:** Docker Compose packaging, frontend dashboard, README +
  demo video, expand golden dataset to 150–300 if time allows.

Cut scope from the back of this list first if time runs short — the eval
harness (milestone 4) is non-negotiable; the frontend (milestone 6) is the
most cuttable.

## 10. Definition of Done (portfolio-ready)

- [ ] End-to-end claim submission → structured report, via API and UI
- [ ] All 5 agent nodes implemented against contracts in §6
- [ ] Golden dataset of ≥150 claims with eval report checked into repo
- [ ] CI running eval on every PR with documented gate thresholds
- [ ] Langfuse trace viewable for any claim_id
- [ ] Documented cost-per-claim before/after model routing
- [ ] README with architecture diagram, demo GIF/video, and metrics table
- [ ] `DECISIONS.md` log of key design tradeoffs