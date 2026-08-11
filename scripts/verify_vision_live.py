#!/usr/bin/env python3
"""Live Vision Agent calibration check against fixtures/images/real/.

Requires docker compose stack already running. Does not start services.
Reads expectations.json, submits each photo via POST /claims, polls until
completed/failed, compares vision output, and writes RESULTS.md.

Exit code 1 if any image is FLAG'd (informational gate; not wired into CI).
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "fixtures" / "images" / "real"
EXPECTATIONS_PATH = REAL_DIR / "expectations.json"
RESULTS_PATH = REAL_DIR / "RESULTS.md"
POLICY_PDF = ROOT / "fixtures" / "sample_policy.pdf"
ESTIMATE_PDF = ROOT / "fixtures" / "sample_estimate.pdf"

API_BASE = "http://localhost:8000"
# Per-claim poll window. HF vision models (OwlViT + CLIP + BLIP) routinely
# need several minutes on first load and tens of seconds when warm; 30s is
# below real inference latency, so the default is 10 minutes.
POLL_TIMEOUT_S = 900.0
POLL_INTERVAL_S = 3.0
# Loud false-positive callout for control photos (matches VISION_LOW_CONFIDENCE_THRESHOLD).
HIGH_CONFIDENCE = 0.4

SEVERITY_RANK = {
    "minor": 0,
    "minor damage": 0,
    "moderate": 1,
    "moderate damage": 1,
    "severe": 2,
    "severe damage": 2,
}


@dataclass
class ImageResult:
    filename: str
    narrative: str
    expected_signal: str
    actual_detections: list[dict[str, Any]] = field(default_factory=list)
    actual_severity: str = ""
    actual_severity_confidence: float | None = None
    actual_vqa: dict[str, str] = field(default_factory=dict)
    status: str = "FLAG"  # PASS | FLAG
    notes: list[str] = field(default_factory=list)
    claim_id: str | None = None
    claim_error: str | None = None
    is_control: bool = False
    high_conf_false_positives: list[dict[str, Any]] = field(default_factory=list)
    upload_note: str | None = None


def _severity_rank(value: str | None) -> int | None:
    if value is None:
        return None
    return SEVERITY_RANK.get(value.strip().lower())


def _format_detections(dets: list[dict[str, Any]]) -> str:
    if not dets:
        return "_(none)_"
    parts = [f"{d.get('label', '?')} ({float(d.get('confidence', 0)):.3f})" for d in dets]
    return ", ".join(parts)


def _format_vqa(vqa: dict[str, str]) -> str:
    if not vqa:
        return "_(none)_"
    return "<br>".join(f"{q}: **{a}**" for q, a in vqa.items())


def _format_expected(exp: dict[str, Any]) -> str:
    bits: list[str] = []
    labels = exp.get("expected_detections_any_of") or []
    if labels:
        bits.append("any of: " + ", ".join(f"`{l}`" for l in labels))
    else:
        bits.append("no damage detections")
    if "expected_severity_min" in exp:
        bits.append(f"severity ≥ `{exp['expected_severity_min']}`")
    if "expected_severity_max" in exp:
        bits.append(f"severity ≤ `{exp['expected_severity_max']}`")
    if exp.get("note"):
        bits.append(exp["note"])
    return "; ".join(bits)


def check_api_reachable(client: httpx.Client) -> None:
    try:
        r = client.get(f"{API_BASE}/health", timeout=5.0)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(
            "ERROR: ClaimSight API is not reachable at "
            f"{API_BASE}/health.\n"
            "Start the stack with `docker compose up` and re-run this script.\n"
            f"Detail: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)


def _ensure_jpeg_for_upload(image_path: Path) -> tuple[Path, str | None, Path | None]:
    """Return (path_to_upload, note, temp_path_to_cleanup).

    API magic-byte checks require real JPEG/PNG. Some fixture files are WebP
    (or other) with a .jpg extension — convert via PIL for upload only.
    """
    header = image_path.read_bytes()[:12]
    is_jpeg = header.startswith(b"\xff\xd8\xff")
    is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
    if is_jpeg or is_png:
        return image_path, None, None

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        rgb.save(tmp_path, format="JPEG", quality=95)

    note = (
        f"source file is not JPEG/PNG magic (got {header[:4]!r}, "
        f"PIL format may differ); converted to temp JPEG for API upload"
    )
    return tmp_path, note, tmp_path


def submit_claim(
    client: httpx.Client,
    *,
    image_path: Path,
    narrative: str,
) -> tuple[str, str | None]:
    upload_path, upload_note, tmp_path = _ensure_jpeg_for_upload(image_path)
    try:
        with (
            POLICY_PDF.open("rb") as policy_f,
            ESTIMATE_PDF.open("rb") as estimate_f,
            upload_path.open("rb") as photo_f,
        ):
            files = {
                "policy_pdf": (POLICY_PDF.name, policy_f, "application/pdf"),
                "estimate_pdf": (ESTIMATE_PDF.name, estimate_f, "application/pdf"),
                "damage_photos": (image_path.name, photo_f, "image/jpeg"),
            }
            data = {"narrative": narrative}
            r = client.post(f"{API_BASE}/claims", files=files, data=data, timeout=60.0)
            if r.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{r.status_code} {r.reason_phrase}: {r.text[:500]}",
                    request=r.request,
                    response=r,
                )
            body = r.json()
            return str(body["claim_id"]), upload_note
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def poll_claim(client: httpx.Client, claim_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        r = client.get(f"{API_BASE}/claims/{claim_id}", timeout=10.0)
        r.raise_for_status()
        last = r.json()
        status = last.get("status")
        if status in {"completed", "failed"}:
            return last
        time.sleep(POLL_INTERVAL_S)
    # Timed out — return last snapshot if any, else synthesize a failed shape.
    if last is not None:
        last = dict(last)
        last["_poll_timeout"] = True
        return last
    return {
        "claim_id": claim_id,
        "status": "failed",
        "result": {"error": f"poll timed out after {POLL_TIMEOUT_S:.0f}s with no response"},
        "_poll_timeout": True,
    }


def evaluate(filename: str, exp: dict[str, Any], claim: dict[str, Any]) -> ImageResult:
    narrative = str(exp.get("narrative") or "")
    expected_labels: list[str] = list(exp.get("expected_detections_any_of") or [])
    is_control = (
        "undamaged" in filename.lower()
        or "control" in filename.lower()
        or (not expected_labels and "false-positive" in str(exp.get("note") or "").lower())
    )

    result = ImageResult(
        filename=filename,
        narrative=narrative,
        expected_signal=_format_expected(exp),
        is_control=is_control,
        claim_id=str(claim.get("claim_id") or ""),
    )

    if claim.get("_poll_timeout") and claim.get("status") not in {"completed", "failed"}:
        result.status = "FLAG"
        result.notes.append(
            f"poll timed out after {POLL_TIMEOUT_S:.0f}s "
            f"(last status={claim.get('status')!r})"
        )
        return result

    status = claim.get("status")
    payload = claim.get("result") or {}

    if status == "failed":
        result.status = "FLAG"
        err = payload.get("error") if isinstance(payload, dict) else None
        result.claim_error = str(err or "claim failed with no error detail")
        result.notes.append(f"claim failed: {result.claim_error}")
        return result

    if status != "completed":
        result.status = "FLAG"
        result.notes.append(f"unexpected claim status={status!r}")
        return result

    vision = payload.get("vision") if isinstance(payload, dict) else None
    if vision is None:
        result.status = "FLAG"
        result.notes.append("result.vision is null")
        return result

    dets = list(vision.get("detections") or [])
    result.actual_detections = dets
    result.actual_severity = str(vision.get("severity_tier") or "")
    conf = vision.get("severity_confidence")
    result.actual_severity_confidence = float(conf) if conf is not None else None
    result.actual_vqa = dict(vision.get("vqa_answers") or {})

    flags: list[str] = []

    # Detection expectations
    actual_labels = {str(d.get("label") or "").strip().lower() for d in dets}
    if expected_labels:
        wanted = {l.strip().lower() for l in expected_labels}
        if not (actual_labels & wanted):
            flags.append(
                "no expected detection label matched "
                f"(wanted any of {sorted(wanted)}; got {sorted(actual_labels) or 'none'})"
            )
    else:
        # Control / empty expectation: any detection is a false-positive signal.
        if dets:
            flags.append(
                f"control/empty expectation but got detections: {_format_detections(dets)}"
            )
        high = [
            d
            for d in dets
            if float(d.get("confidence") or 0.0) >= HIGH_CONFIDENCE
        ]
        result.high_conf_false_positives = high

    # Severity min / max
    actual_rank = _severity_rank(result.actual_severity)
    if "expected_severity_min" in exp:
        min_rank = _severity_rank(str(exp["expected_severity_min"]))
        if actual_rank is None or min_rank is None or actual_rank < min_rank:
            flags.append(
                f"severity {result.actual_severity!r} below "
                f"expected_severity_min={exp['expected_severity_min']!r}"
            )
    if "expected_severity_max" in exp:
        max_rank = _severity_rank(str(exp["expected_severity_max"]))
        if actual_rank is None or max_rank is None or actual_rank > max_rank:
            flags.append(
                f"severity {result.actual_severity!r} above "
                f"expected_severity_max={exp['expected_severity_max']!r}"
            )

    if flags:
        result.status = "FLAG"
        result.notes.extend(flags)
    else:
        result.status = "PASS"

    return result


def summarize_flag_patterns(results: list[ImageResult]) -> str:
    flagged = [r for r in results if r.status == "FLAG"]
    if not flagged:
        return "No FLAGs — all images met the expectation checks."

    # Lightweight pattern notes from flag reasons / filenames.
    reasons: list[str] = []
    severity_flags = [r for r in flagged if any("severity" in n for n in r.notes)]
    detection_miss = [
        r
        for r in flagged
        if any("no expected detection" in n for n in r.notes)
    ]
    false_pos = [
        r
        for r in flagged
        if any("control/empty" in n or "detections" in n.lower() for n in r.notes)
        and r.is_control
    ]
    failures = [r for r in flagged if r.claim_error or any("failed" in n for n in r.notes)]
    timeouts = [r for r in flagged if any("timed out" in n for n in r.notes)]

    if timeouts:
        reasons.append(
            f"{len(timeouts)} image(s) timed out waiting for claim completion "
            f"({POLL_TIMEOUT_S:.0f}s) — may indicate cold model load or slow inference."
        )
    if failures and not timeouts:
        reasons.append(
            f"{len(failures)} image(s) returned claim status=failed "
            "(pipeline error, not a vision calibration miss)."
        )
    if detection_miss:
        names = ", ".join(r.filename for r in detection_miss)
        reasons.append(
            f"{len(detection_miss)} image(s) missed expected detection labels "
            f"({names}) — vocabulary miss or scores below the 0.15 floor."
        )
    if severity_flags:
        names = ", ".join(r.filename for r in severity_flags)
        reasons.append(
            f"{len(severity_flags)} image(s) had severity out of expected range ({names})."
        )
    if false_pos:
        reasons.append(
            f"{len(false_pos)} control/undamaged photo(s) produced damage detections "
            "(false-positive signal)."
        )

    # Low-confidence noise on control
    for r in results:
        if r.is_control and r.actual_detections:
            scores = [float(d.get("confidence") or 0) for d in r.actual_detections]
            if scores and max(scores) < HIGH_CONFIDENCE:
                reasons.append(
                    f"Control photo {r.filename} has detections only below "
                    f"{HIGH_CONFIDENCE} — confidence floor of 0.15 may admit noise."
                )

    if not reasons:
        detail = "; ".join(
            f"{r.filename}: {', '.join(r.notes) or 'unspecified'}" for r in flagged
        )
        reasons.append(f"FLAG details: {detail}")

    return " ".join(reasons)


def render_report(results: list[ImageResult]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pass_n = sum(1 for r in results if r.status == "PASS")
    flag_n = sum(1 for r in results if r.status == "FLAG")
    total = len(results)

    lines: list[str] = [
        "# Vision Agent — live real-photo verification",
        "",
        f"_Auto-generated by `scripts/verify_vision_live.py` at {now}. "
        "Calibration check only — PASS/FLAG is not a correctness test._",
        "",
        "## Method notes",
        "",
        f"- Poll timeout per claim: **{POLL_TIMEOUT_S:.0f}s** (extended beyond a "
        "naive 30s window because OwlViT+CLIP+BLIP cold/warm CPU inference exceeds that).",
        f"- High-confidence false-positive bar for control callout: "
        f"**≥ {HIGH_CONFIDENCE}** (same as `VISION_LOW_CONFIDENCE_THRESHOLD`).",
    ]
    upload_noted = [r for r in results if r.upload_note]
    if upload_noted:
        lines.append(
            "- Upload conversions: "
            + "; ".join(f"`{r.filename}` — {r.upload_note}" for r in upload_noted)
        )
    else:
        lines.append("- Upload conversions: none (all sources were JPEG/PNG magic).")
    lines.extend(
        [
            "",
            "## Per-image results",
            "",
            "| Filename | Narrative | Expected signal | Actual detections | Actual severity | Actual VQA | Result |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    for r in results:
        sev = r.actual_severity or "—"
        if r.actual_severity_confidence is not None:
            sev = f"{sev} ({r.actual_severity_confidence:.3f})"
        note_suffix = ""
        if r.notes:
            note_suffix = "<br>_" + "; ".join(r.notes) + "_"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{r.filename}`",
                    r.narrative.replace("|", "\\|"),
                    r.expected_signal.replace("|", "\\|"),
                    _format_detections(r.actual_detections).replace("|", "\\|"),
                    sev.replace("|", "\\|"),
                    _format_vqa(r.actual_vqa).replace("|", "\\|"),
                    f"**{r.status}**{note_suffix}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- **Total images tested:** {total}",
            f"- **PASS:** {pass_n}",
            f"- **FLAG:** {flag_n}",
            f"- **Pattern note:** {summarize_flag_patterns(results)}",
            "",
        ]
    )

    # Explicit false-positive callout for control / undamaged photos
    controls = [r for r in results if r.is_control]
    lines.append("## Control / undamaged false-positive callout")
    lines.append("")
    if not controls:
        lines.append("_No control/undamaged photo present in this run._")
    else:
        any_high = False
        for r in controls:
            high = r.high_conf_false_positives or [
                d
                for d in r.actual_detections
                if float(d.get("confidence") or 0.0) >= HIGH_CONFIDENCE
            ]
            if high:
                any_high = True
                lines.append(
                    f"**ALERT:** `{r.filename}` produced high-confidence "
                    f"(≥ {HIGH_CONFIDENCE}) damage detection(s): "
                    f"{_format_detections(high)}. "
                    "This is the primary false-positive signal for calibration."
                )
            elif r.actual_detections:
                lines.append(
                    f"**Note:** `{r.filename}` produced damage detection(s) below "
                    f"the high-confidence bar ({HIGH_CONFIDENCE}): "
                    f"{_format_detections(r.actual_detections)}. "
                    "Still a false-positive at the 0.15 floor — worth watching."
                )
            elif r.claim_error or any("timed out" in n or "failed" in n for n in r.notes):
                lines.append(
                    f"**Could not evaluate** `{r.filename}` "
                    f"({'failed/timeout'}: {'; '.join(r.notes)})."
                )
            else:
                lines.append(
                    f"`{r.filename}`: no damage detections — control check clean."
                )
        if not any_high and any(
            r.actual_detections and not r.claim_error for r in controls
        ):
            lines.append("")
            lines.append(
                "_No high-confidence (≥ 0.4) false positives on control, "
                "but lower-score detections above the 0.15 floor may still indicate noise._"
            )
        elif not any_high and all(not r.actual_detections for r in controls):
            lines.append("")
            lines.append(
                "No high-confidence false-positive damage detections on control photos."
            )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not EXPECTATIONS_PATH.is_file():
        print(f"ERROR: missing expectations manifest: {EXPECTATIONS_PATH}", file=sys.stderr)
        return 2
    if not POLICY_PDF.is_file() or not ESTIMATE_PDF.is_file():
        print(
            f"ERROR: sample PDFs missing ({POLICY_PDF.name}, {ESTIMATE_PDF.name})",
            file=sys.stderr,
        )
        return 2

    expectations: dict[str, Any] = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    if not expectations:
        print("ERROR: expectations.json is empty", file=sys.stderr)
        return 2

    with httpx.Client() as client:
        check_api_reachable(client)

        results: list[ImageResult] = []
        for filename, exp in expectations.items():
            image_path = REAL_DIR / filename
            print(f"--- {filename} ---", flush=True)
            if not image_path.is_file():
                r = ImageResult(
                    filename=filename,
                    narrative=str(exp.get("narrative") or ""),
                    expected_signal=_format_expected(exp),
                    status="FLAG",
                    notes=[f"image file not found: {image_path}"],
                    is_control="undamaged" in filename.lower(),
                )
                results.append(r)
                print(f"  FLAG: file missing", flush=True)
                continue

            try:
                claim_id, upload_note = submit_claim(
                    client, image_path=image_path, narrative=str(exp.get("narrative") or "")
                )
                print(f"  claim_id={claim_id}", flush=True)
                if upload_note:
                    print(f"  upload_note: {upload_note}", flush=True)
                claim = poll_claim(client, claim_id)
                evaluated = evaluate(filename, exp, claim)
                evaluated.upload_note = upload_note
                if upload_note:
                    evaluated.notes.append(upload_note)
                results.append(evaluated)
                print(
                    f"  status={claim.get('status')} -> {evaluated.status}"
                    + (f" ({'; '.join(evaluated.notes)})" if evaluated.notes else ""),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 — never crash the whole run
                r = ImageResult(
                    filename=filename,
                    narrative=str(exp.get("narrative") or ""),
                    expected_signal=_format_expected(exp),
                    status="FLAG",
                    notes=[f"script error: {exc}"],
                    is_control="undamaged" in filename.lower(),
                )
                results.append(r)
                print(f"  FLAG: {exc}", flush=True)

    report = render_report(results)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"\nWrote {RESULTS_PATH}", flush=True)

    if any(r.status == "FLAG" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
