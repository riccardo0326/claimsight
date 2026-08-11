# Handoff — Vision Agent live real-photo verification

**Date stuck:** 2026-08-07  
**Prior chats:** [Vision agent verification report](1f5d31ae-0655-43e3-916f-3967c9747675), [Vision Agent slice](7f0361a1-71b0-4908-9d91-14ab8ea2d230)  
**Goal (unchanged):** Measure Slice 3 Vision Agent on real photos via the live docker compose API. **Do not modify agent logic** (`agents/vision_agent.py`, thresholds, label sets). Measurement only. If findings look bad, stop and report — no fixes in this task.

---

## Status summary

| Item | Status |
|---|---|
| Slice 3 Vision Agent code + automated tests | Done (prior chat); synthetic fixtures only |
| Real photos in `fixtures/images/real/` | Present (gitignored `*.jpg`) |
| `fixtures/images/real/expectations.json` | **Done** |
| `scripts/verify_vision_live.py` | **Done** (with a few resume fixes still wise) |
| Live run producing meaningful `RESULTS.md` | **Not done** |
| Calibration read / go-no-go for Adjudicator | **Not done** |
| Agent logic changes | None (correct — out of scope) |

**Current `RESULTS.md` is not usable.** It only records poll timeouts / Windows console encoding errors / a dropped connection — **no real `result.vision` data.** Delete or overwrite it on the next successful run.

**Docker Desktop was down** when this handoff was written (`npipe:////./pipe/dockerDesktopLinuxEngine` missing). Start Docker before resuming.

---

## What the original prompt asked for

1. Expectation manifest → `fixtures/images/real/expectations.json`
2. Verification script → `scripts/verify_vision_live.py` (API reachable check; POST `/claims` + poll; PASS/FLAG; exit 1 if any FLAG)
3. Auto-generated report → `fixtures/images/real/RESULTS.md`
4. Do **not** tweak vision agent / thresholds / expectations after seeing results
5. When done: paste full `RESULTS.md` + one-paragraph read on whether confidence floor `0.15` and candidate labels look reasonable

---

## Done

### Files created

| Path | Notes |
|---|---|
| `fixtures/images/real/expectations.json` | One entry per real photo; inferred from filenames (user said ask if unclear — filenames were clear). |
| `scripts/verify_vision_live.py` | Full harness: health check, submit, poll, evaluate, write `RESULTS.md`, print report, exit 1 on FLAG. |
| `docker-compose.verify.yml` | Temp override: mounts `./:/app` into worker so Slice 3 code is live without image rebuild. Keep HF on named volume `hf_cache`. |
| `scripts/_seed_hf_cache.sh` | Helper (optional cleanup) — copy host HF models into docker volume. |
| `scripts/_prefetch_minilm.py` | Helper (optional cleanup) — prefetch MiniLM into a container. |
| `fixtures/images/real/RESULTS.md` | **Stale / invalid** — regenerate. |

### Real photos present

```
front_collision.jpg          # WEBP bytes with .jpg name (RIFF magic) — script converts to temp JPEG for API
side_damage.jpg              # real JPEG
shattered_windshield.jpg     # real JPEG
inside_airbag_exploded.jpg   # real JPEG
totaled_car.jpg              # real JPEG
undamaged_car.jpg            # real JPEG (control / false-positive check)
```

### Script behavior already implemented

- Fails clearly if `http://localhost:8000/health` is down (do not start compose yourself from the script).
- Submits `fixtures/sample_policy.pdf` + `fixtures/sample_estimate.pdf` + one photo + narrative from manifest.
- Poll timeout currently **900s** (task said 30s; that is below real HF CPU latency — keep 900s or higher for cold start).
- Converts non-JPEG magic (WebP) to temp JPEG before upload.
- PASS/FLAG (not fail); records detections, severity, VQA verbatim; control false-positive callout at ≥ 0.4.
- Exit code 1 if any FLAG.

### Infra progress before PC freeze

- Running worker image **predated Slice 3** (`ModuleNotFoundError: agents.vision_agent`). Mitigations tried: `docker cp` agents into container; later `docker-compose.verify.yml` bind-mount `./:/app`.
- Full `docker compose build` of api/worker was started (Dockerfile prefetches OwlViT/CLIP/BLIP) but was **killed** as too slow; not required if source is mounted.
- HF cache volume `claimsight_hf_cache` had incomplete BLIP (`.incomplete` blob). Complete BLIP was copied from host:
  - Host: `%USERPROFILE%\.cache\huggingface\hub\models--Salesforce--blip-vqa-base` (~1.5GB, complete)
  - Volume after copy: BLIP present with **0** `.incomplete` files
- Volume also had: OwlViT, CLIP, LayoutLM. **MiniLM embeddings may still be missing** from the volume (RAG needs them after Vision).
- Worker got as far as loading DocVQA → OwlViT → CLIP → starting BLIP; first claims stayed `pending` for 10–20+ minutes (CPU-bound + HF Hub timeouts).
- Redis was `FLUSHALL`'d once to clear a backlog of stale tasks.

---

## Not done

1. **Successful end-to-end verification run** with completed claims and real `result.vision` per image.
2. **Valid `RESULTS.md`** (table + summary + control callout with actual scores).
3. **Human calibration read** for the user (0.15 floor + label vocabulary go/no-go before Adjudicator).
4. **Stable docker worker** with Slice 3 code + complete HF weights + enough RAM (Docker Desktop was ~3.7GiB limit; loading DocVQA + OwlViT + CLIP + BLIP is tight).
5. Optional cleanup: delete `scripts/_seed_hf_cache.sh`, `scripts/_prefetch_minilm.py`, `docker-compose.verify.yml` after a proper image rebuild — or keep verify override until `docker compose build` lands Slice 3 in the image.
6. Optional: re-encode `front_collision.jpg` as real JPEG on disk (fixture hygiene); script already converts at upload time.

---

## Known pitfalls (read before re-running)

1. **Worker image may lack Slice 3 code** unless you rebuild **or** use:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.verify.yml up -d worker
   ```
2. **Named volume `hf_cache` mounts over image-baked weights** — Dockerfile prefetch does not help at runtime unless the volume is seeded.
3. **First claim cold-start is very slow** (10–30+ min on CPU if models load/download). Warm subsequent claims are faster but still tens of seconds each. Poll timeout must be ≥ 900s for the first image after restart.
4. **Windows console `charmap` bug:** an earlier print used `→` and crashed after evaluate, double-counting FLAGs. Script was patched to use `->`; also run with:
   ```powershell
   $env:PYTHONUTF8 = "1"
   python scripts/verify_vision_live.py
   ```
5. **`front_collision.jpg` is WebP** labeled `.jpg` — API returns 422 without conversion (script handles this).
6. **Do not change** `expectations.json` after seeing real results (prompt rule).
7. **Do not change** vision thresholds/labels in this task even if results look wrong — stop and report.

---

## Resume checklist (new chat)

Paste this into the new agent (or open this file):

```
Resume ClaimSight Vision live verification from docs/VISION_LIVE_VERIFY_HANDOFF.md.
Do not modify agents/vision_agent.py. Get a real RESULTS.md via scripts/verify_vision_live.py
against docker compose, then give me RESULTS.md + calibration read.
```

### Steps

1. **Start Docker Desktop**, wait until the engine is up.
2. From repo root:
   ```powershell
   docker compose -f docker-compose.yml -f docker-compose.verify.yml up -d
   ```
   Confirm: `GET http://localhost:8000/health` → ok; worker has vision:
   ```powershell
   docker compose exec worker python -c "from agents.vision_agent import DETECTION_LABELS; print(DETECTION_LABELS)"
   ```
3. **Seed MiniLM if RAG fails** (only if claims fail on missing embedding model):
   - Prefetch `sentence-transformers/all-MiniLM-L6-v2` into `claimsight_hf_cache` / `/cache/huggingface`.
4. **Optional warm-up:** submit one claim manually and wait until worker logs show OwlViT + CLIP + BLIP loaded and claim `completed`.
5. **Run verification:**
   ```powershell
   $env:PYTHONUTF8 = "1"
   python scripts/verify_vision_live.py
   ```
6. Confirm `fixtures/images/real/RESULTS.md` has real detections/severity/VQA (not just timeouts).
7. Deliver to user:
   - Full contents of `RESULTS.md`
   - One paragraph: are floor `0.15` and label vocab reasonable, or does this need a follow-up slice before Adjudicator?

### If stack is still too slow / OOM

- Increase Docker Desktop memory (8GB+ recommended for this stack).
- Or rebuild images once (`docker compose build api worker`) so Slice 3 is baked in, then drop the verify override.
- Still do **not** change agent logic; only get a clean measurement.

---

## Acceptance criteria (still open)

- [ ] Script runs against live compose without crashing on individual claim failures
- [ ] `RESULTS.md` has per-image table + summary + control false-positive callout with **actual vision output**
- [ ] Exit non-zero if any FLAG (informational gate; not CI)
- [ ] User gets RESULTS + calibration paragraph
- [ ] No changes to vision agent / thresholds / post-hoc expectation edits

---

## Out of scope reminder

This task is **measurement only**. A follow-up slice (if needed) would handle calibration (threshold, labels, false positives on `undamaged_car.jpg`, severity misses, etc.).
