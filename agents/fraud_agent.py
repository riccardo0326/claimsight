"""Fraud/Risk Agent — zero-shot narrative signal + deterministic cross-checks (Slice 4).

Contract: RiskOutput in agents/schemas.py. Heuristic score documented in DECISIONS.md.
Classification is a SIGNAL, not proof of fraud.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agents.schemas import DocumentOutput, RiskFlag, RiskOutput, VerifierOutput
from api.config import get_settings

logger = logging.getLogger(__name__)

CANDIDATE_LABELS = [
    "consistent claim",
    "inconsistent claim",
    "possible staged damage",
    "weather mismatch",
    "recall-related damage",
]

# Narrative tokens that imply weather/storm as the claimed cause.
_WEATHER_CAUSE_RE = re.compile(
    r"\b(hail|storm|thunderstorm|tornado|hurricane|flood|flooding|"
    r"wind\s*damage|weather|ice\s*storm|blizzard)\b",
    re.IGNORECASE,
)

# Severity ladder (portfolio heuristic — not calibrated).
SEVERITY_INFO = "info"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"

# Risk-score weights (sum of positive contributions, then clamp to [0, 1]).
# Avoid double-counting: rule flags take precedence over matching classifier labels.
WEIGHT_INCONSISTENT = 0.35
WEIGHT_STAGED = 0.40
WEIGHT_WEATHER_RULE = 0.30
WEIGHT_RECALL_RULE = 0.10
WEIGHT_CLASSIFIER_WEATHER = 0.15
WEIGHT_CLASSIFIER_RECALL = 0.05
CLASSIFIER_SCORE_FLOOR = 0.45

_zero_shot_pipeline: Any | None = None


def _get_zero_shot_pipeline() -> Any:
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        from transformers import pipeline

        settings = get_settings()
        logger.info("Loading fraud zero-shot model %s", settings.fraud_zero_shot_model)
        _zero_shot_pipeline = pipeline(
            "zero-shot-classification",
            model=settings.fraud_zero_shot_model,
        )
    return _zero_shot_pipeline


def _classify_narrative(narrative: str) -> dict[str, float]:
    """Return label → score map. Empty narrative → empty scores (graceful)."""
    text = (narrative or "").strip()
    if not text:
        return {}
    pipe = _get_zero_shot_pipeline()
    result = pipe(text, candidate_labels=CANDIDATE_LABELS, multi_label=True)
    labels = result.get("labels") or []
    scores = result.get("scores") or []
    return {str(lab): float(score) for lab, score in zip(labels, scores, strict=False)}


def _damage_context(document: DocumentOutput, narrative: str) -> str:
    parts = [narrative or ""]
    for item in document.line_items:
        parts.append(item.description)
    return " ".join(parts).lower()


def _weather_mismatch_flag(
    narrative: str,
    verifiers: VerifierOutput,
) -> RiskFlag | None:
    if not _WEATHER_CAUSE_RE.search(narrative or ""):
        return None
    weather = verifiers.weather_at_incident
    if weather is None:
        return None
    if weather.had_storm_event:
        return None
    return RiskFlag(
        flag_type="weather mismatch",
        rationale=(
            "Narrative cites weather/storm damage, but weather_at_incident."
            f"had_storm_event is False (condition={weather.condition!r}, "
            f"precipitation_mm={weather.precipitation_mm})."
        ),
        severity=SEVERITY_MEDIUM,
    )


def _recall_related_flag(
    document: DocumentOutput,
    narrative: str,
    verifiers: VerifierOutput,
) -> RiskFlag | None:
    if not verifiers.nhtsa_recalls:
        return None
    context = _damage_context(document, narrative)
    if not context.strip():
        return None
    for recall in verifiers.nhtsa_recalls:
        component = (recall.component or "").strip().lower()
        if not component or component == "unknown":
            continue
        # Token overlap: component tokens (len>=4) against claim context, with a
        # light plural stem so "bumpers" matches "bumper".
        tokens = [t for t in re.split(r"[^a-z0-9]+", component) if len(t) >= 4]
        matched = False
        for tok in tokens:
            stem = tok.rstrip("s")
            if tok in context or (len(stem) >= 4 and stem in context):
                matched = True
                break
        if matched:
            return RiskFlag(
                flag_type="recall-related damage",
                rationale=(
                    f"NHTSA recall {recall.campaign_number} component "
                    f"{recall.component!r} overlaps claimed damage context. "
                    "Informational only — a recall does not prove fraud."
                ),
                severity=SEVERITY_INFO,
            )
    return None


def _classifier_flags(scores: dict[str, float]) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    inconsistent = scores.get("inconsistent claim", 0.0)
    staged = scores.get("possible staged damage", 0.0)
    if inconsistent >= CLASSIFIER_SCORE_FLOOR:
        flags.append(
            RiskFlag(
                flag_type="inconsistent claim",
                rationale=(
                    f"Zero-shot classifier scored 'inconsistent claim' at "
                    f"{inconsistent:.2f} (signal only, not proof)."
                ),
                severity=SEVERITY_MEDIUM if inconsistent < 0.7 else SEVERITY_HIGH,
            )
        )
    if staged >= CLASSIFIER_SCORE_FLOOR:
        flags.append(
            RiskFlag(
                flag_type="possible staged damage",
                rationale=(
                    f"Zero-shot classifier scored 'possible staged damage' at "
                    f"{staged:.2f} (signal only, not proof)."
                ),
                severity=SEVERITY_HIGH,
            )
        )
    return flags


def compute_risk_score(
    *,
    flags: list[RiskFlag],
    classifier_scores: dict[str, float],
) -> float:
    """Deterministic weighted heuristic in [0, 1]. See DECISIONS.md D20.

    Rule flags and classifier labels for the same evidence family are not both
    fully counted: weather/recall rule contributions suppress the matching
    classifier contribution.
    """
    flag_types = {f.flag_type for f in flags}
    score = 0.0

    if "possible staged damage" in flag_types:
        score += WEIGHT_STAGED
    elif classifier_scores.get("possible staged damage", 0.0) >= CLASSIFIER_SCORE_FLOOR:
        score += WEIGHT_STAGED * classifier_scores["possible staged damage"]

    if "inconsistent claim" in flag_types:
        score += WEIGHT_INCONSISTENT
    elif classifier_scores.get("inconsistent claim", 0.0) >= CLASSIFIER_SCORE_FLOOR:
        score += WEIGHT_INCONSISTENT * classifier_scores["inconsistent claim"]

    if "weather mismatch" in flag_types:
        score += WEIGHT_WEATHER_RULE
    elif classifier_scores.get("weather mismatch", 0.0) >= CLASSIFIER_SCORE_FLOOR:
        score += WEIGHT_CLASSIFIER_WEATHER * classifier_scores["weather mismatch"]

    if "recall-related damage" in flag_types:
        score += WEIGHT_RECALL_RULE
    elif classifier_scores.get("recall-related damage", 0.0) >= CLASSIFIER_SCORE_FLOOR:
        score += WEIGHT_CLASSIFIER_RECALL * classifier_scores["recall-related damage"]

    return max(0.0, min(1.0, round(score, 4)))


def run_fraud_agent(
    narrative: str,
    document: DocumentOutput,
    verifiers: VerifierOutput,
    *,
    classifier_scores: dict[str, float] | None = None,
) -> RiskOutput:
    """Run Fraud/Risk Agent. Inject classifier_scores in tests to skip HF load."""
    scores = (
        classifier_scores
        if classifier_scores is not None
        else _classify_narrative(narrative)
    )

    flags: list[RiskFlag] = []
    weather_flag = _weather_mismatch_flag(narrative, verifiers)
    if weather_flag is not None:
        flags.append(weather_flag)
    recall_flag = _recall_related_flag(document, narrative, verifiers)
    if recall_flag is not None:
        flags.append(recall_flag)
    flags.extend(_classifier_flags(scores))

    risk_score = compute_risk_score(flags=flags, classifier_scores=scores)
    return RiskOutput(flags=flags, risk_score=risk_score)
