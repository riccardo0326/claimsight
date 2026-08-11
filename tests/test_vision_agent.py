"""Unit tests for the Vision Agent.

Default suite: empty-input path (no HF load).
`pytest -m hf`: real zero-shot models on synthetic fixtures (CPU).
"""

from __future__ import annotations

import pytest
from PIL import Image

from agents.schemas import Detection, VisionOutput
from agents.vision_agent import (
    DETECTION_LABELS,
    VQA_QUESTIONS,
    _owlvit_prompt,
    run_vision_agent,
)
from api.config import Settings, get_settings

# Locked by DECISIONS.md D14 — scene/part vocabulary after OWL-ViT calibration.
EXPECTED_DETECTION_LABELS = [
    "damaged car",
    "crashed car",
    "shattered windshield",
    "broken headlight",
    "deployed airbag",
    "broken glass",
]


def _stub_severity_and_vqa(monkeypatch) -> None:
    """Avoid loading CLIP/BLIP while testing detection plumbing."""
    monkeypatch.setattr(
        "agents.vision_agent._classify_severity",
        lambda _image: ("minor damage", 0.9),
    )
    monkeypatch.setattr(
        "agents.vision_agent._vqa_one",
        lambda _image, _question: ("no", 0.9),
    )


def _mock_detection_pipe(monkeypatch, mock_results: list[dict]) -> list[dict]:
    """Replace OWL-ViT pipeline; always return fixed results (ignore HF threshold)."""
    calls: list[dict] = []

    def fake_pipe(_image, candidate_labels=None, threshold=None):
        calls.append(
            {
                "candidate_labels": list(candidate_labels or []),
                "threshold": threshold,
            }
        )
        return mock_results

    monkeypatch.setattr(
        "agents.vision_agent._get_detection_pipeline",
        lambda: fake_pipe,
    )
    return calls


@pytest.fixture
def vision_photo(tmp_path):
    path = tmp_path / "damage.jpg"
    Image.new("RGB", (64, 64), color=(40, 80, 120)).save(path, format="JPEG")
    return path


def test_empty_image_paths_returns_none():
    assert run_vision_agent([]) is None


def test_vision_detection_threshold_constant():
    # DECISIONS.md D14: floor sits above observed undamaged FP peak (~0.43).
    assert Settings.model_fields["vision_detection_threshold"].default == 0.45


def test_detection_label_vocabulary_matches_d14():
    assert DETECTION_LABELS == EXPECTED_DETECTION_LABELS


def test_detection_threshold_enforcement(monkeypatch, vision_photo):
    """Keep scores at/above the calibrated floor; drop those below.

    Boundary is inclusive (`score >= threshold`): 0.45 is kept, 0.44 is not.
    """
    _stub_severity_and_vqa(monkeypatch)
    get_settings.cache_clear()
    # Force the calibrated default even if a local .env overrides it.
    monkeypatch.setenv("VISION_DETECTION_THRESHOLD", "0.45")
    get_settings.cache_clear()

    # Fixed OWL-ViT-shaped rows (templated labels, as the real pipeline returns).
    mock_results = [
        {"label": "a photo of a broken glass", "score": 0.30},
        {"label": "a photo of a broken headlight", "score": 0.44},
        {"label": "a photo of a shattered windshield", "score": 0.45},
        {"label": "a photo of a crashed car", "score": 0.46},
        {"label": "a photo of a damaged car", "score": 0.60},
    ]
    _mock_detection_pipe(monkeypatch, mock_results)

    output = run_vision_agent([str(vision_photo)])
    assert output is not None
    kept = {(d.label, round(d.confidence, 2)) for d in output.detections}
    # Inclusive floor: 0.45, 0.46, 0.60 kept; 0.30 and 0.44 excluded.
    assert kept == {
        ("shattered windshield", 0.45),
        ("crashed car", 0.46),
        ("damaged car", 0.60),
    }
    assert all(d.confidence >= 0.45 for d in output.detections)
    assert "broken glass" not in {d.label for d in output.detections}
    assert "broken headlight" not in {d.label for d in output.detections}


def test_owlvit_prompt_template_applied_and_stripped(monkeypatch, vision_photo):
    _stub_severity_and_vqa(monkeypatch)
    monkeypatch.setenv("VISION_DETECTION_THRESHOLD", "0.45")
    get_settings.cache_clear()

    calls = _mock_detection_pipe(
        monkeypatch,
        [{"label": "a photo of a damaged car", "score": 0.55}],
    )

    output = run_vision_agent([str(vision_photo)])
    assert output is not None
    assert len(calls) == 1

    expected_prompts = [_owlvit_prompt(lab) for lab in EXPECTED_DETECTION_LABELS]
    assert calls[0]["candidate_labels"] == expected_prompts
    # Article choice: consonant -> "a", vowel-initial -> "an".
    assert "a photo of a damaged car" in expected_prompts
    assert "a photo of a deployed airbag" in expected_prompts  # 'd' -> "a"
    assert all(p.startswith("a photo of ") for p in calls[0]["candidate_labels"])

    assert len(output.detections) == 1
    assert output.detections[0].label == "damaged car"
    assert not output.detections[0].label.startswith("a photo of")


@pytest.mark.hf
def test_vision_agent_synthetic_images(synthetic_images):
    """Structural smoke test — do not assert specific detection labels."""
    paths = [str(p) for p in synthetic_images]
    output = run_vision_agent(paths)
    assert isinstance(output, VisionOutput)
    assert isinstance(output.detections, list)
    for det in output.detections:
        assert isinstance(det, Detection)
        assert det.label
        assert 0.0 <= det.confidence <= 1.0
        assert det.image_path in paths
    assert isinstance(output.severity_tier, str)
    assert output.severity_tier
    assert isinstance(output.severity_confidence, float)
    assert 0.0 <= output.severity_confidence <= 1.0
    assert isinstance(output.low_confidence, bool)
    assert output.low_confidence == (output.severity_confidence < 0.4)


@pytest.mark.hf
def test_vision_output_schema_fields(synthetic_images):
    output = run_vision_agent([str(p) for p in synthetic_images])
    assert output is not None
    # Round-trip through Pydantic to prove schema shape.
    dumped = output.model_dump(mode="json")
    reloaded = VisionOutput.model_validate(dumped)
    assert set(reloaded.vqa_answers.keys()) == set(VQA_QUESTIONS)
    for answer in reloaded.vqa_answers.values():
        assert answer in {"yes", "no", "unknown"}
    for det in reloaded.detections:
        assert {"label", "confidence", "image_path"} <= set(det.model_dump().keys())
