#!/usr/bin/env python3
"""Diagnose OWL-ViT zero-shot detection on fixtures/images/real/.

Runs the detection model DIRECTLY (not via vision_agent._detect_one) so we can
see raw pre-threshold scores, compare prompt phrasing, and inspect the exact
tensors fed to the model.

Usage (from repo root):
  python scripts/debug_owlvit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "fixtures" / "images" / "real"
OUT_DIR = ROOT / "scripts" / "debug_output"
MODEL_ID = "google/owlvit-base-patch32"

# Current production labels (agents/vision_agent.py). Still compares bare vs
# templated ("a photo of a …") prompt phrasing.
DETECTION_LABELS = [
    "damaged car",
    "crashed car",
    "shattered windshield",
    "broken headlight",
    "deployed airbag",
    "broken glass",
]

# Alternate vocabularies to probe for a fix (not production). Hypothesis F/G/H.
ALT_LABEL_SETS: dict[str, list[str]] = {
    "concrete_nouns": [
        "car",
        "headlight",
        "windshield",
        "airbag",
        "bumper",
        "glass",
    ],
    "damage_compounds": [
        "crashed car",
        "damaged car",
        "shattered windshield",
        "broken headlight",
        "deployed airbag",
        "crumpled bumper",
        "dent in car",
        "scratch on car",
    ],
    "owlvit_style": [
        "a photo of a car",
        "a photo of a damaged car",
        "a photo of a crashed car",
        "a photo of a headlight",
        "a photo of a broken headlight",
        "a photo of a windshield",
        "a photo of a shattered windshield",
        "a photo of an airbag",
        "a photo of a deployed airbag",
        "a photo of a bumper",
        "a photo of a dent",
        "a photo of broken glass",
    ],
}

# Production detection floor (logged for comparison only — we do NOT filter).
PROD_THRESHOLD = 0.45


def _magic_header(path: Path) -> bytes:
    return path.read_bytes()[:12]


def _describe_image(path: Path) -> dict[str, Any]:
    header = _magic_header(path)
    with Image.open(path) as img:
        info = {
            "filename": path.name,
            "file_bytes": path.stat().st_size,
            "magic4": list(header[:4]),
            "magic4_repr": repr(header[:4]),
            "pil_format": img.format,
            "pil_mode_raw": img.mode,
            "pil_size_raw": list(img.size),
        }
        rgb = img.convert("RGB")
        rgb.load()
        arr = np.asarray(rgb)
        info.update(
            {
                "pil_mode_rgb": rgb.mode,
                "pil_size_rgb": list(rgb.size),
                "array_shape": list(arr.shape),
                "array_dtype": str(arr.dtype),
                "pixel_min": int(arr.min()),
                "pixel_max": int(arr.max()),
                "pixel_mean": float(arr.mean()),
                "is_all_black": bool(arr.max() == 0),
                "is_all_white": bool(arr.min() == 255),
            }
        )
        return info, rgb


def _save_model_input(
    *,
    stem: str,
    rgb: Image.Image,
    pixel_values: Any,
) -> dict[str, str]:
    """Persist the RGB image and a visualizable view of model pixel_values."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rgb_path = OUT_DIR / f"{stem}_rgb.png"
    rgb.save(rgb_path)

    # pixel_values typically [1, 3, H, W], roughly ImageNet-normalized.
    pv = pixel_values.detach().cpu().float().numpy()
    if pv.ndim == 4:
        pv = pv[0]
    # CHW -> HWC, un-normalize with common CLIP/OWL-ViT mean/std for inspection.
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)
    unnorm = np.clip(pv * std + mean, 0.0, 1.0)
    hwc = (np.transpose(unnorm, (1, 2, 0)) * 255.0).astype(np.uint8)
    tensor_path = OUT_DIR / f"{stem}_model_input.png"
    Image.fromarray(hwc, mode="RGB").save(tensor_path)

    npy_path = OUT_DIR / f"{stem}_pixel_values.npy"
    np.save(npy_path, pixel_values.detach().cpu().numpy())

    return {
        "rgb_png": str(rgb_path.relative_to(ROOT)),
        "model_input_png": str(tensor_path.relative_to(ROOT)),
        "pixel_values_npy": str(npy_path.relative_to(ROOT)),
    }


def _score_stats(scores: list[float]) -> dict[str, float | int | None]:
    if not scores:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "above_0_05": 0,
            "above_0_10": 0,
            "above_0_15": 0,
            "above_0_20": 0,
        }
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "above_0_05": int((arr >= 0.05).sum()),
        "above_0_10": int((arr >= 0.10).sum()),
        "above_0_15": int((arr >= 0.15).sum()),
        "above_0_20": int((arr >= 0.20).sum()),
    }


def _summarize_detections(
    results: list[dict[str, Any]],
    labels: list[str],
    *,
    pass_name: str,
) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = {lab: [] for lab in labels}
    all_scores: list[float] = []
    for item in results:
        label = str(item.get("label") or "").strip()
        score = float(item.get("score") or 0.0)
        box = item.get("box")
        all_scores.append(score)
        entry = {"score": score, "box": box}
        if label in by_label:
            by_label[label].append(entry)
        else:
            by_label.setdefault(label, []).append(entry)

    per_label_summary: dict[str, Any] = {}
    for lab in labels:
        scores = [e["score"] for e in by_label.get(lab, [])]
        top = sorted(by_label.get(lab, []), key=lambda e: e["score"], reverse=True)[:5]
        per_label_summary[lab] = {
            "raw_count": len(scores),
            "max_score": float(max(scores)) if scores else None,
            "top5": top,
        }

    return {
        "pass_name": pass_name,
        "labels": labels,
        "raw_triple_count": len(results),
        "score_stats": _score_stats(all_scores),
        "per_label": per_label_summary,
        "would_pass_prod_threshold_0_15": [
            {
                "label": str(item.get("label")),
                "score": float(item.get("score") or 0.0),
                "box": item.get("box"),
            }
            for item in sorted(
                results, key=lambda x: float(x.get("score") or 0.0), reverse=True
            )
            if float(item.get("score") or 0.0) >= PROD_THRESHOLD
        ][:20],
    }


def _run_pipeline_pass(
    pipe: Any,
    image: Image.Image,
    labels: list[str],
    *,
    pass_name: str,
) -> dict[str, Any]:
    """Call zero-shot pipeline with threshold=0.0 to keep all raw detections."""
    # Explicit threshold=0.0 — HF pipeline default (~0.1) would hide low scores.
    results = pipe(image, candidate_labels=labels, threshold=0.0)
    if results is None:
        results = []
    return _summarize_detections(results, labels, pass_name=pass_name)


def _run_direct_forward_pass(
    model: Any,
    processor: Any,
    image: Image.Image,
    labels: list[str],
    *,
    pass_name: str,
) -> dict[str, Any]:
    """Bypass pipeline postprocess; decode with threshold=0.0 via processor."""
    import torch

    # Nested list: batch size 1 with N text queries (OwlViT processor convention).
    inputs = processor(text=[labels], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # logits: [batch, num_patches, num_queries]; keep every box/query score.
    target_sizes = torch.tensor([image.size[::-1]])  # (h, w)
    decoded = processor.post_process_object_detection(
        outputs=outputs,
        threshold=0.0,
        target_sizes=target_sizes,
    )[0]

    results: list[dict[str, Any]] = []
    scores = decoded["scores"]
    labels_idx = decoded["labels"]
    boxes = decoded["boxes"]
    for score, lab_i, box in zip(scores, labels_idx, boxes, strict=False):
        li = int(lab_i)
        label = labels[li] if 0 <= li < len(labels) else f"idx:{li}"
        box_list = [float(x) for x in box.tolist()]
        results.append(
            {
                "label": label,
                "score": float(score),
                "box": {
                    "xmin": box_list[0],
                    "ymin": box_list[1],
                    "xmax": box_list[2],
                    "ymax": box_list[3],
                },
            }
        )

    # Also capture raw max sigmoid logits per query (before box filtering).
    logits = outputs.logits[0]  # [num_patches, num_queries]
    probs = torch.sigmoid(logits)
    per_query_max = probs.max(dim=0).values.tolist()
    summary = _summarize_detections(results, labels, pass_name=pass_name)
    summary["raw_logit_sigmoid_max_per_query"] = {
        labels[i]: float(per_query_max[i]) for i in range(len(labels))
    }
    summary["raw_logit_sigmoid_global_max"] = float(probs.max())
    return summary


def _print_pass(summary: dict[str, Any]) -> None:
    stats = summary["score_stats"]
    print(f"\n  === Pass: {summary['pass_name']} ===")
    print(f"  Labels: {summary['labels']}")
    print(f"  Raw (label, box, score) triples (pre-filter): {summary['raw_triple_count']}")
    print(
        "  Score stats: "
        f"min={stats['min']}, max={stats['max']}, mean={stats['mean']}, "
        f"p50={stats['p50']}, p90={stats['p90']}, p99={stats['p99']}"
    )
    print(
        "  Counts >= thresholds: "
        f">=0.05:{stats['above_0_05']}  >=0.10:{stats['above_0_10']}  "
        f">=0.15:{stats['above_0_15']}  >=0.20:{stats['above_0_20']}"
    )
    print("  Per-label max raw score:")
    for lab, info in summary["per_label"].items():
        mx = info["max_score"]
        mx_s = f"{mx:.6f}" if mx is not None else "None (0 boxes)"
        print(f"    - {lab!r}: max={mx_s}  raw_boxes={info['raw_count']}")
    survivors = summary["would_pass_prod_threshold_0_15"]
    if survivors:
        print(f"  Would survive prod floor {PROD_THRESHOLD}:")
        for s in survivors[:10]:
            print(f"    - {s['label']!r} score={s['score']:.6f} box={s['box']}")
    else:
        print(f"  Would survive prod floor {PROD_THRESHOLD}: (none)")
    if "raw_logit_sigmoid_global_max" in summary:
        print(
            f"  Direct forward global max sigmoid(logit): "
            f"{summary['raw_logit_sigmoid_global_max']:.6f}"
        )
        print("  Per-query max sigmoid(logit) across all patches:")
        for lab, val in summary.get("raw_logit_sigmoid_max_per_query", {}).items():
            print(f"    - {lab!r}: {val:.6f}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-alt",
        action="store_true",
        help="Only probe alternate label sets (skip production-label passes).",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))

    image_paths = sorted(REAL_DIR.glob("*.jpg")) + sorted(REAL_DIR.glob("*.png"))
    if not image_paths:
        print(f"ERROR: no images in {REAL_DIR}", file=sys.stderr)
        return 1

    print(f"Loading model {MODEL_ID} ...")
    from transformers import OwlViTForObjectDetection, OwlViTProcessor, pipeline

    processor = OwlViTProcessor.from_pretrained(MODEL_ID)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_ID)
    model.eval()

    if args.probe_alt:
        return _run_alt_label_probe(model, processor, image_paths)

    # Match production: pipeline(model=<hub id>). Passing a preloaded processor
    # alone makes transformers fail to resolve the tokenizer on some versions.
    pipe = pipeline("zero-shot-object-detection", model=MODEL_ID)

    # Inspect pipeline default threshold (hypothesis: default filters everything).
    default_threshold = getattr(pipe, "threshold", None)
    call_defaults = {}
    try:
        import inspect

        sig = inspect.signature(pipe.__call__)
        for name, param in sig.parameters.items():
            if name == "threshold":
                call_defaults["threshold_default"] = param.default
    except Exception as exc:  # noqa: BLE001
        call_defaults["inspect_error"] = str(exc)

    print(f"Pipeline threshold attr: {default_threshold!r}")
    print(f"__call__ threshold default: {call_defaults}")

    templated_labels = [f"a photo of a {lab}" for lab in DETECTION_LABELS]
    report: list[dict[str, Any]] = []

    for path in image_paths:
        print("\n" + "=" * 72)
        print(f"IMAGE: {path.name}")
        info, rgb = _describe_image(path)
        print(
            f"  file={info['file_bytes']}B magic={info['magic4_repr']} "
            f"PIL format={info['pil_format']} mode={info['pil_mode_raw']}->"
            f"{info['pil_mode_rgb']} size={info['pil_size_raw']}"
        )
        print(
            f"  array shape={info['array_shape']} dtype={info['array_dtype']} "
            f"min/max/mean={info['pixel_min']}/{info['pixel_max']}/{info['pixel_mean']:.2f} "
            f"blank={info['is_all_black'] or info['is_all_white']}"
        )

        # Exact tensors the model receives (processor path used by OwlViT).
        text_inputs = processor(
            text=[DETECTION_LABELS],
            images=rgb,
            return_tensors="pt",
        )
        saved = _save_model_input(
            stem=path.stem,
            rgb=rgb,
            pixel_values=text_inputs["pixel_values"],
        )
        pv = text_inputs["pixel_values"]
        print(
            f"  model pixel_values shape={tuple(pv.shape)} "
            f"dtype={pv.dtype} min/max/mean="
            f"{float(pv.min()):.4f}/{float(pv.max()):.4f}/{float(pv.mean()):.4f}"
        )
        print(f"  saved: {saved}")

        # Pass A: bare labels (same as production candidate set) via pipeline.
        bare = _run_pipeline_pass(pipe, rgb, DETECTION_LABELS, pass_name="bare_labels_pipeline")
        _print_pass(bare)

        # Pass A2: same bare labels via direct model forward (true pre-filter).
        bare_direct = _run_direct_forward_pass(
            model, processor, rgb, DETECTION_LABELS, pass_name="bare_labels_direct"
        )
        _print_pass(bare_direct)

        # Pass B: OWL-ViT recommended prompt template.
        templated = _run_pipeline_pass(
            pipe, rgb, templated_labels, pass_name="templated_a_photo_of_a_pipeline"
        )
        _print_pass(templated)

        templated_direct = _run_direct_forward_pass(
            model,
            processor,
            rgb,
            templated_labels,
            pass_name="templated_a_photo_of_a_direct",
        )
        _print_pass(templated_direct)

        # Also probe default-threshold pipeline behavior (what vision_agent gets).
        default_results = pipe(rgb, candidate_labels=DETECTION_LABELS)
        default_count = 0 if default_results is None else len(default_results)
        default_max = None
        if default_results:
            default_max = max(float(x.get("score") or 0.0) for x in default_results)
        print(
            f"\n  Default-threshold pipeline call (as vision_agent): "
            f"count={default_count} max_score={default_max}"
        )

        report.append(
            {
                "image": info,
                "saved": saved,
                "bare_labels_pipeline": bare,
                "bare_labels_direct": bare_direct,
                "templated_pipeline": templated,
                "templated_direct": templated_direct,
                "default_threshold_call": {
                    "count": default_count,
                    "max_score": default_max,
                },
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "owlvit_raw_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"Wrote JSON report: {report_path}")
    print(f"Wrote visual dumps under: {OUT_DIR}")
    return 0


def _run_alt_label_probe(model: Any, processor: Any, image_paths: list[Path]) -> int:
    """Probe alternate vocabularies — still diagnosis only, no production writes."""
    report: list[dict[str, Any]] = []
    print("ALT LABEL PROBE — production DETECTION_LABELS unchanged")
    for path in image_paths:
        print("\n" + "=" * 72)
        print(f"IMAGE: {path.name}")
        info, rgb = _describe_image(path)
        print(
            f"  size={info['pil_size_rgb']} format={info['pil_format']} "
            f"mean={info['pixel_mean']:.1f}"
        )
        image_entry: dict[str, Any] = {"image": path.name, "sets": {}}
        for set_name, labels in ALT_LABEL_SETS.items():
            summary = _run_direct_forward_pass(
                model, processor, rgb, labels, pass_name=f"alt_{set_name}"
            )
            _print_pass(summary)
            image_entry["sets"][set_name] = {
                "global_max": summary.get("raw_logit_sigmoid_global_max"),
                "per_label_max": {
                    k: v["max_score"] for k, v in summary["per_label"].items()
                },
                "survivors_0_15": summary["would_pass_prod_threshold_0_15"][:10],
            }
        report.append(image_entry)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "owlvit_alt_labels_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"Wrote alt-label report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
