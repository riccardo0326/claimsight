"""Unit tests for the Vision Agent.

Default suite: empty-input path (no HF load).
`pytest -m hf`: real zero-shot models on synthetic fixtures (CPU).
"""

from __future__ import annotations

import pytest

from agents.schemas import Detection, VisionOutput
from agents.vision_agent import VQA_QUESTIONS, run_vision_agent


def test_empty_image_paths_returns_none():
    assert run_vision_agent([]) is None


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
