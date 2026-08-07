"""End-to-end integration: POST claim → eager Celery → GET completed result."""

from __future__ import annotations

import time

import pytest

from agents.schemas import VisionOutput


def test_submit_and_poll_until_completed(client, policy_pdf, estimate_pdf, expected):
    files = {
        "policy_pdf": ("sample_policy.pdf", policy_pdf.read_bytes(), "application/pdf"),
        "estimate_pdf": ("sample_estimate.pdf", estimate_pdf.read_bytes(), "application/pdf"),
    }
    create = client.post("/claims", files=files)
    assert create.status_code == 202, create.text
    body = create.json()
    assert body["status"] == "pending"
    claim_id = body["claim_id"]

    # Eager Celery runs inline on delay(), so the claim should already be terminal.
    # Still poll with a short bound in case of timing quirks.
    result = None
    status = None
    for _ in range(20):
        resp = client.get(f"/claims/{claim_id}")
        assert resp.status_code == 200
        payload = resp.json()
        status = payload["status"]
        result = payload["result"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert status == "completed", f"status={status} result={result}"
    assert result is not None
    doc = result["document_agent"]
    assert doc["policy_id"] == expected["policy_id"]
    assert doc["coverage_limits"] == expected["coverage_limits"]
    assert doc["deductible"] == expected["deductible"]
    assert doc["vin"] == expected["vin"]
    assert doc["incident_date"] == expected["incident_date"]
    assert doc["line_items"] == expected["line_items"]
    assert "extraction_meta" in result
    assert "rag" in result
    # No damage photos → vision is null (not an empty object).
    assert result.get("vision") is None


@pytest.mark.hf
def test_submit_with_damage_photos_includes_vision(
    vision_client, policy_pdf, estimate_pdf, synthetic_images
):
    files = [
        ("policy_pdf", ("sample_policy.pdf", policy_pdf.read_bytes(), "application/pdf")),
        ("estimate_pdf", ("sample_estimate.pdf", estimate_pdf.read_bytes(), "application/pdf")),
    ]
    for path in synthetic_images:
        files.append(
            ("damage_photos", (path.name, path.read_bytes(), "image/jpeg")),
        )

    create = vision_client.post("/claims", files=files)
    assert create.status_code == 202, create.text
    claim_id = create.json()["claim_id"]

    result = None
    status = None
    for _ in range(20):
        resp = vision_client.get(f"/claims/{claim_id}")
        assert resp.status_code == 200
        payload = resp.json()
        status = payload["status"]
        result = payload["result"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert status == "completed", f"status={status} result={result}"
    assert result is not None
    assert result.get("vision") is not None
    vision = VisionOutput.model_validate(result["vision"])
    assert isinstance(vision.low_confidence, bool)
    assert isinstance(vision.detections, list)
    assert isinstance(vision.vqa_answers, dict)


def test_submit_with_narrative_retrieves_scoped_clauses(
    rag_client, policy_pdf, estimate_pdf, expected, sample_clauses
):
    sample_ids = {c["clause_id"] for c in sample_clauses["clauses"]}
    narrative = (
        "Vehicle was in a front-end collision. Insured requests collision coverage "
        "review against the policy for bumper and headlight repairs."
    )
    files = {
        "policy_pdf": ("sample_policy.pdf", policy_pdf.read_bytes(), "application/pdf"),
        "estimate_pdf": ("sample_estimate.pdf", estimate_pdf.read_bytes(), "application/pdf"),
    }
    data = {"narrative": narrative}
    create = rag_client.post("/claims", files=files, data=data)
    assert create.status_code == 202, create.text
    claim_id = create.json()["claim_id"]

    result = None
    status = None
    for _ in range(20):
        resp = rag_client.get(f"/claims/{claim_id}")
        assert resp.status_code == 200
        payload = resp.json()
        status = payload["status"]
        result = payload["result"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert status == "completed", f"status={status} result={result}"
    assert result is not None
    rag = result["rag"]
    clauses = rag["retrieved_clauses"]
    assert clauses, "expected non-empty retrieved_clauses"
    for clause in clauses:
        assert clause["clause_id"] in sample_ids
        assert "similarity_score" in clause
        assert clause["text"]


def test_get_unknown_claim_returns_404(client):
    resp = client.get("/claims/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
