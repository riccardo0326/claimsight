#!/usr/bin/env python3
"""Post-fix: run calibrated vision_agent._detect_one on fixtures/images/real/.

Skips CLIP/VQA so verification is fast. Writes NDJSON via vision_agent instrumentation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.vision_agent import DETECTION_LABELS, _detect_one, _owlvit_prompt  # noqa: E402
from api.config import get_settings  # noqa: E402

REAL_DIR = ROOT / "fixtures" / "images" / "real"


def main() -> int:
    settings = get_settings()
    # Drop cached settings if an old .env still has 0.15.
    threshold = settings.vision_detection_threshold
    print(f"DETECTION_LABELS={DETECTION_LABELS}")
    print(f"threshold={threshold}")
    print(f"sample prompts={[ _owlvit_prompt(l) for l in DETECTION_LABELS ]}")

    summary = []
    for path in sorted(REAL_DIR.glob("*.jpg")) + sorted(REAL_DIR.glob("*.png")):
        with Image.open(path) as img:
            image = img.convert("RGB")
            image.load()
        dets = _detect_one(image, str(path), threshold)
        row = {
            "image": path.name,
            "count": len(dets),
            "detections": [
                {"label": d.label, "confidence": d.confidence} for d in dets
            ],
        }
        summary.append(row)
        print(f"{path.name}: {row['detections'] or '(none)'}")

    out = ROOT / "scripts" / "debug_output" / "post_fix_detections.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
