"""Vision Agent — zero-shot damage signal from claim photos.

Contract: VisionOutput in agents/schemas.py. Models and aggregation choices
are logged in DECISIONS.md (D14).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from agents.schemas import Detection, VisionOutput
from api.config import get_settings

logger = logging.getLogger(__name__)

DETECTION_LABELS = [
    "dent",
    "scratch",
    "broken glass",
    "broken headlight",
    "deployed airbag",
    "bumper damage",
]
SEVERITY_LABELS = [
    "minor damage",
    "moderate damage",
    "severe damage",
]
SEVERITY_RANK = {
    "minor damage": 0,
    "moderate damage": 1,
    "severe damage": 2,
}
VQA_QUESTIONS = [
    "Is the airbag deployed?",
    "Is there broken glass visible?",
    "Is the vehicle drivable?",
]
MAX_DETECTIONS_PER_IMAGE = 5
VQA_UNKNOWN_SCORE_FLOOR = 0.1

_detection_pipeline: Any | None = None
_classification_pipeline: Any | None = None
_vqa_processor: Any | None = None
_vqa_model: Any | None = None


def _get_detection_pipeline() -> Any:
    global _detection_pipeline
    if _detection_pipeline is None:
        from transformers import pipeline

        settings = get_settings()
        logger.info("Loading zero-shot detection model %s", settings.vision_detection_model)
        _detection_pipeline = pipeline(
            "zero-shot-object-detection",
            model=settings.vision_detection_model,
        )
    return _detection_pipeline


def _get_classification_pipeline() -> Any:
    global _classification_pipeline
    if _classification_pipeline is None:
        from transformers import pipeline

        settings = get_settings()
        logger.info(
            "Loading zero-shot classification model %s",
            settings.vision_classification_model,
        )
        _classification_pipeline = pipeline(
            "zero-shot-image-classification",
            model=settings.vision_classification_model,
        )
    return _classification_pipeline


def _get_vqa_model() -> tuple[Any, Any]:
    """Load BLIP VQA via processor/model (not the removed pipeline alias).

    transformers>=5 dropped `pipeline(\"visual-question-answering\")`; Blip
    processor + BlipForQuestionAnswering works on both 4.x and 5.x.
    """
    global _vqa_processor, _vqa_model
    if _vqa_processor is None or _vqa_model is None:
        from transformers import BlipForQuestionAnswering, BlipProcessor

        settings = get_settings()
        logger.info("Loading VQA model %s", settings.vision_vqa_model)
        _vqa_processor = BlipProcessor.from_pretrained(settings.vision_vqa_model)
        _vqa_model = BlipForQuestionAnswering.from_pretrained(settings.vision_vqa_model)
        _vqa_model.eval()
    return _vqa_processor, _vqa_model


def _normalize_yes_no(raw: str | None) -> str:
    if raw is None:
        return "unknown"
    text = str(raw).strip().lower()
    if not text:
        return "unknown"
    # Prefer leading yes/no tokens common in VQA free-text answers.
    if text.startswith("yes") or text in {"y", "true"}:
        return "yes"
    if text.startswith("no") or text in {"n", "false"}:
        return "no"
    if "yes" in text and "no" not in text:
        return "yes"
    if "no" in text and "yes" not in text:
        return "no"
    return "unknown"


def _detect_one(image: Image.Image, image_path: str, threshold: float) -> list[Detection]:
    pipe = _get_detection_pipeline()
    results = pipe(image, candidate_labels=DETECTION_LABELS)
    if not results:
        return []

    scored: list[tuple[str, float]] = []
    for item in results:
        label = str(item.get("label") or "").strip()
        score = float(item.get("score") or 0.0)
        if label and score >= threshold:
            scored.append((label, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        Detection(label=label, confidence=score, image_path=image_path)
        for label, score in scored[:MAX_DETECTIONS_PER_IMAGE]
    ]


def _classify_severity(image: Image.Image) -> tuple[str, float]:
    pipe = _get_classification_pipeline()
    results = pipe(image, candidate_labels=SEVERITY_LABELS)
    if not results:
        return "minor damage", 0.0

    best = results[0]
    label = str(best.get("label") or "minor damage").strip().lower()
    score = float(best.get("score") or 0.0)
    if label not in SEVERITY_RANK:
        label = "minor damage"
    return label, score


def _vqa_one(image: Image.Image, question: str) -> tuple[str, float]:
    import torch

    processor, model = _get_vqa_model()
    inputs = processor(image, question, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            return_dict_in_generate=True,
            output_scores=True,
        )
    raw = processor.decode(outputs.sequences[0], skip_special_tokens=True)
    answer = _normalize_yes_no(raw)

    # Approximate confidence from first generated-token softmax max.
    score = 0.0
    if outputs.scores:
        probs = torch.softmax(outputs.scores[0], dim=-1)
        score = float(probs.max().item())

    if score < VQA_UNKNOWN_SCORE_FLOOR:
        return "unknown", score
    return answer, score


def run_vision_agent(image_paths: list[str]) -> VisionOutput | None:
    """Extract damage detections, severity, and VQA answers from claim photos.

    Returns None when image_paths is empty (no model load). Missing or unreadable
    paths are skipped with a warning.
    """
    if not image_paths:
        return None

    settings = get_settings()
    threshold = settings.vision_detection_threshold
    low_conf_threshold = settings.vision_low_confidence_threshold

    all_detections: list[Detection] = []
    severity_candidates: list[tuple[str, float]] = []
    # question -> list of (answer, score)
    vqa_candidates: dict[str, list[tuple[str, float]]] = {q: [] for q in VQA_QUESTIONS}

    for path_str in image_paths:
        path = Path(path_str)
        if not path.is_file():
            logger.warning("Vision Agent: image not found, skipping: %s", path_str)
            continue
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
                image.load()
        except OSError as exc:
            logger.warning("Vision Agent: failed to open %s: %s", path_str, exc)
            continue

        all_detections.extend(_detect_one(image, path_str, threshold))
        severity_candidates.append(_classify_severity(image))
        for question in VQA_QUESTIONS:
            vqa_candidates[question].append(_vqa_one(image, question))

    # Max-severity-wins: highest tier across images; confidence from that image.
    if severity_candidates:
        best_tier, best_score = max(
            severity_candidates,
            key=lambda pair: (SEVERITY_RANK.get(pair[0], -1), pair[1]),
        )
        severity_tier = best_tier
        severity_confidence = best_score
    else:
        # All paths unreadable — still return a valid empty-ish VisionOutput.
        severity_tier = "minor damage"
        severity_confidence = 0.0

    vqa_answers: dict[str, str] = {}
    for question, answers in vqa_candidates.items():
        if not answers:
            vqa_answers[question] = "unknown"
            continue
        answer, _score = max(answers, key=lambda pair: pair[1])
        vqa_answers[question] = answer

    return VisionOutput(
        detections=all_detections,
        severity_tier=severity_tier,
        severity_confidence=severity_confidence,
        vqa_answers=vqa_answers,
        low_confidence=severity_confidence < low_conf_threshold,
    )
