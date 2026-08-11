# Handoff — Vision Agent live real-photo verification

**Status:** **Closed** (2026-08-11) — post-D14 calibration live run succeeded.  
**Prior goal:** Measure Slice 3 Vision Agent on real photos via the live docker compose API.

---

## Final status

| Item | Status |
|---|---|
| Slice 3 Vision Agent + D14 calibration | **Done** (templated OWL-ViT prompts, scene/part labels, floor 0.45) |
| Mocked regression tests (`tests/test_vision_agent.py`) | **Done** |
| Real photos in `fixtures/images/real/` | Present |
| `expectations.json` | Aligned to D14 labels; `side_damage` documented OWL-ViT miss |
| `scripts/verify_vision_live.py` | Done (console ASCII-safe print; RESULTS.md stays UTF-8) |
| Live `RESULTS.md` (post-calibration) | **Done** — regenerated 2026-08-11 15:45:23 UTC |
| Detection calibration go/no-go | **Go** for Adjudicator (scene labels fire; control clean) |
| Severity calibration | **Soft miss** — CLIP still often returns `minor damage` on moderate scenes; not blocking Slice 3 close |

### Latest live result (after worker restart)

- **PASS 2 / FLAG 4** (exit 1 = informational FLAGs)
- Detections non-empty on damage photos except known `side_damage` miss
- `totaled_car` + `undamaged_car` PASS
- All 4 FLAGs are **severity below expected_severity_min**, not empty OWL-ViT output
- Control: no damage detections

**Ops note:** After changing `vision_agent.py` / config defaults, **restart the Celery worker** — bind-mounted source does not reload already-imported modules in a long-lived worker. A fresh `docker exec ... python -c` can look calibrated while Celery still runs old code.

---

## Slice 3 close read

OWL-ViT detection is calibrated and regression-guarded (D14). Remaining FLAGs are CLIP severity under-calling “moderate” on several real photos — a follow-up if Adjudicator needs tighter severity, not a reopen of detection vocabulary/floor. VQA remains usable (e.g. airbag **yes** on cabin photo).

Optional cleanup later: rebuild images and drop `docker-compose.verify.yml` / seed helpers; re-encode `front_collision.jpg` as real JPEG on disk.
