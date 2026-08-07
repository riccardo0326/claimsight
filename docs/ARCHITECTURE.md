# ClaimSight — Architecture

## 1. System Overview

ClaimSight ingests a multimodal insurance claim (damage photos, policy PDF,
repair estimate, police report, claim narrative) and produces a structured,
citation-grounded coverage/fraud-risk recommendation. The system is built as
a directed multi-agent graph (LangGraph) sitting behind an async job API,
with a first-class evaluation harness and observability layer — not a
single prompt-in/response-out wrapper.

```
                        ┌─────────────────────────┐
                        │     FastAPI Gateway      │
                        │  POST /claims  (submit)  │
                        │  GET  /claims/{id}       │
                        └───────────┬──────────────┘
                                    │ enqueue
                                    ▼
                        ┌─────────────────────────┐
                        │   Redis + Celery Queue   │
                        └───────────┬──────────────┘
                                    │ worker picks up job
                                    ▼
                     ┌──────────────────────────────┐
                     │      LangGraph Orchestrator    │
                     │  (state machine, see §3)       │
                     └───────┬───────┬───────┬────────┘
                             │       │       │
              ┌──────────────┘       │       └──────────────┐
              ▼                      ▼                       ▼
     ┌─────────────────┐   ┌─────────────────┐    ┌─────────────────────┐
     │  Vision Agent     │   │ Document Agent   │    │   RAG Agent          │
     │  (detection,      │   │ (DocQA/TableQA   │    │ (pgvector retrieval  │
     │  classification,  │   │  field extract)  │    │  over policy clauses │
     │  VQA)              │   │                  │    │  + precedent claims) │
     └────────┬──────────┘   └────────┬─────────┘    └──────────┬───────────┘
              │                       │                          │
              └───────────┬───────────┴─────────────┬────────────┘
                           ▼                         ▼
                  ┌──────────────────┐     ┌────────────────────┐
                  │  Fraud/Risk Agent │     │  External Verifiers │
                  │  (zero-shot +     │     │  NHTSA VIN/Recalls, │
                  │   text classify)  │     │  NOAA weather        │
                  └─────────┬─────────┘     └──────────┬──────────┘
                            └───────────┬───────────────┘
                                        ▼
                             ┌────────────────────┐
                             │  Adjudicator Agent   │
                             │ (synthesis + cited    │
                             │  decision + confidence)│
                             └──────────┬────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │  Structured Claim Report    │
                          │  (JSON: decision, citations, │
                          │   confidence, flags)         │
                          └──────────────┬────────────────┘
                                         │
                    ┌────────────────────┼─────────────────────┐
                    ▼                    ▼                     ▼
           ┌────────────────┐  ┌──────────────────┐  ┌──────────────────┐
           │ Postgres (state) │  │ Langfuse (traces) │  │ Eval Harness (CI) │
           └────────────────┘  └──────────────────┘  └──────────────────┘
                    │
                    ▼
           ┌────────────────────┐
           │  Frontend Dashboard  │
           │  (claim view, agent   │
           │   trace, citations)   │
           └────────────────────┘
```

## 2. Components

### 2.1 Ingestion Layer
- Accepts multipart submission: images (damage photos), PDFs (policy,
  estimate, police report), free-text narrative.
- Files land in object storage (local disk / S3-compatible for portfolio
  purposes); a claim record is created in Postgres with status `pending`.
- Job enqueued to Celery via Redis broker. API returns `claim_id`
  immediately — ingestion is async, never blocking.

### 2.2 LangGraph Orchestrator
- Implemented as an explicit state graph, not a free-form agent loop.
- Shared `ClaimState` object (typed, e.g. Pydantic/TypedDict) is passed
  between nodes and accumulates each agent's output.
- Nodes with no data dependency (Vision, Document, external verifiers) run
  in parallel branches; RAG and Fraud/Risk nodes depend on Document Agent
  output (need extracted fields before retrieval/scoring); Adjudicator is
  the single terminal node.
- Explicit graph (vs. a general ReAct agent) is a deliberate choice:
  deterministic control flow, cheaper to debug, and each node is
  independently unit-testable.

### 2.3 Agent Nodes

| Node | HF Task(s) | Input | Output |
|---|---|---|---|
| Vision Agent | Object Detection, Image Classification, VQA | damage photos | bounding boxes, severity tier, VQA answers (e.g. airbag deployed) |
| Document Agent | Document Question Answering, Table Question Answering | policy PDF, estimate PDF | structured fields: coverage limits, deductible, VIN, dates, line-item costs |
| RAG Agent | Feature Extraction, Sentence Similarity | extracted fields + narrative | top-k retrieved policy clauses + precedent claims with similarity scores |
| Fraud/Risk Agent | Zero-Shot Classification, Text Classification | narrative + external verifier results | risk flags with rationale (e.g. "reported hail damage, no storm event on record") |
| External Verifiers | — (plain API calls, not HF tasks) | VIN, date, location | NHTSA recall/complaint matches, NOAA weather history for claim date/location |
| Adjudicator | Text Generation | all upstream outputs | final decision (approve/deny/review), confidence score, cited clauses, flags |

### 2.4 RAG Subsystem
- **Store:** Postgres + pgvector. Chosen over a managed vector DB
  deliberately — demonstrates owning the retrieval infra, not just calling
  an API.
- **Ingestion pipeline (LlamaIndex):** policy PDFs are chunked (semantic or
  fixed-window with overlap), embedded via a sentence-similarity model, and
  written to a `policy_clauses` table with metadata (policy_id, clause_type,
  page number).
- **Retrieval:** hybrid — vector similarity + metadata filter (e.g. only
  retrieve clauses from the policy tied to this claim's `policy_id`).
- **Grounding contract:** the Adjudicator is only allowed to cite clause IDs
  that appear in the retrieved set for that claim. Any citation outside
  that set is treated as a hallucination and caught by the guardrail layer
  (§2.6) before the report is finalized.

### 2.5 Adjudicator / Synthesis
- Consumes the full `ClaimState` (all upstream agent outputs).
- Produces a structured JSON report: `decision`, `confidence`,
  `cited_clauses[]`, `risk_flags[]`, `reasoning_summary`.
- Low-confidence or conflicting-signal claims are routed to
  `needs_human_review` rather than forced to a binary decision — this is
  the safety-relevant design choice worth calling out in interviews.

### 2.6 Guardrails
- Citation-validity check: reject/flag any adjudication citing a clause ID
  not present in that claim's retrieved context.
- Schema validation on every agent's structured output (Pydantic) before it
  is written into `ClaimState` — a malformed node output halts that branch
  and routes to review rather than silently propagating bad data.

### 2.7 Eval Harness
- Golden dataset: 150–300 labeled claims (synthetic-but-realistic, clearly
  documented as such) with ground-truth decision, expected citation, and
  fraud-flag label.
- Metrics: groundedness/faithfulness (RAGAS or LLM-judge), citation
  hallucination rate, decision accuracy vs. ground truth, fraud-flag
  precision/recall, latency and cost per claim.
- Runs as a CI job (GitHub Actions) on every PR touching `agents/`,
  `rag/`, or prompts — a regression in any metric blocks merge.

### 2.8 LLMOps / Observability
- Langfuse traces every node invocation: prompt, output, tokens, latency,
  cost — attached to the claim's `claim_id` for full traceability.
- Model routing: small/local model for classification-style nodes (Vision
  labels, Fraud/Risk classification), frontier model reserved for the
  Adjudicator's synthesis step. This routing decision is the basis for the
  cost-per-claim metric in the eval report.
- Prompt versioning: prompts are stored as versioned templates, not inline
  strings, so eval runs can be diffed across prompt versions.

### 2.9 Frontend
- Displays: claim intake summary, per-agent output, retrieved citations
  (with source clause text), final decision with confidence, and the full
  agent trace (for debugging/demo purposes).
- Built last, once real pipeline output exists to design against.

## 3. Data Flow (state machine)

1. `submitted` → ingestion validates files, creates claim record
2. `processing` → orchestrator invokes Vision + Document + External
   Verifier nodes in parallel
3. Document Agent output unblocks RAG Agent and Fraud/Risk Agent
4. All branches join → Adjudicator node runs
5. Guardrail check runs on Adjudicator output
6. `completed` (with decision) or `needs_human_review` (low confidence /
   guardrail failure)
7. Report + full trace persisted; claim record updated; frontend polls or
   subscribes for status

## 4. Scalability Considerations
- Celery workers scale horizontally; vision/document extraction nodes are
  the most compute-heavy and can be scaled independently of lightweight
  classification nodes.
- pgvector indexed with IVFFlat/HNSW for retrieval at scale beyond the
  portfolio dataset size.
- Model routing (§2.8) is the primary lever for cost control at volume —
  documented explicitly with before/after cost-per-claim numbers.
- Stateless API/worker layer — horizontal scaling is a config change, not
  an architecture change.

## 5. Explicit Non-Goals (for scope control)
- Not a real production insurance system — no PII handling compliance
  (HIPAA/SOC2), no real customer data.
- Not optimizing for exhaustive fraud detection — the fraud agent
  demonstrates the pattern (external data cross-referencing), not a
  state-of-the-art fraud model.
- Not building custom-trained vision models from scratch — using
  pretrained/fine-tuned HF models is acceptable and expected.