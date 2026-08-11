"""Integration tests for Slice 4 pipeline (mocked HTTP + mocked fraud classifier)."""

from __future__ import annotations

import time
from datetime import date

import pytest

from agents.schemas import DocumentOutput, RiskOutput, VerifierOutput, WeatherAtIncident
from agents.verifiers import SOURCE_NHTSA_VIN


def test_submit_persists_optional_incident_location(client, policy_pdf, estimate_pdf):
    import uuid

    from db import session as db_session
    from db.models import Claim

    files = {
        "policy_pdf": ("sample_policy.pdf", policy_pdf.read_bytes(), "application/pdf"),
        "estimate_pdf": ("sample_estimate.pdf", estimate_pdf.read_bytes(), "application/pdf"),
    }
    data = {
        "narrative": "Front-end collision.",
        "incident_location": "Austin, TX",
    }
    create = client.post("/claims", files=files, data=data)
    assert create.status_code == 202, create.text
    claim_id = create.json()["claim_id"]

    assert db_session.SessionLocal is not None
    with db_session.SessionLocal() as db:
        claim = db.get(Claim, uuid.UUID(claim_id))
        assert claim is not None
        assert claim.incident_location == "Austin, TX"

    for _ in range(20):
        resp = client.get(f"/claims/{claim_id}")
        payload = resp.json()
        if payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert payload["status"] == "completed"
    assert "verifiers" in payload["result"]
    assert "risk" in payload["result"]


def test_submit_without_incident_location_still_completes(client, policy_pdf, estimate_pdf):
    files = {
        "policy_pdf": ("sample_policy.pdf", policy_pdf.read_bytes(), "application/pdf"),
        "estimate_pdf": ("sample_estimate.pdf", estimate_pdf.read_bytes(), "application/pdf"),
    }
    create = client.post("/claims", files=files)
    assert create.status_code == 202
    claim_id = create.json()["claim_id"]
    resp = client.get(f"/claims/{claim_id}")
    assert resp.json()["status"] == "completed"
    assert resp.json()["result"]["verifiers"]["weather_at_incident"] is None


def test_worker_integration_mocked_verifiers_and_fraud(
    client, policy_pdf, estimate_pdf, monkeypatch
):
    """Full worker path with real run_verifiers/run_fraud_agent but mocked I/O."""

    def fake_verifiers(document, *, incident_location, db, client=None):  # noqa: ANN001
        assert isinstance(document, DocumentOutput)
        return VerifierOutput(
            make="HONDA",
            model="Accord",
            model_year=2003,
            weather_at_incident=WeatherAtIncident(
                condition="Clear",
                precipitation_mm=0.0,
                had_storm_event=False,
            ),
            sources_failed=[],
        )

    def fake_fraud(narrative, document, verifiers, **_kwargs):
        return RiskOutput(flags=[], risk_score=0.12)

    monkeypatch.setattr("worker.tasks.run_verifiers", fake_verifiers)
    monkeypatch.setattr("worker.tasks.run_fraud_agent", fake_fraud)

    files = {
        "policy_pdf": ("sample_policy.pdf", policy_pdf.read_bytes(), "application/pdf"),
        "estimate_pdf": ("sample_estimate.pdf", estimate_pdf.read_bytes(), "application/pdf"),
    }
    data = {"narrative": "Collision claim", "incident_location": "Denver, CO"}
    create = client.post("/claims", files=files, data=data)
    claim_id = create.json()["claim_id"]
    payload = client.get(f"/claims/{claim_id}").json()
    assert payload["status"] == "completed"
    result = payload["result"]
    assert "document_agent" in result
    assert "extraction_meta" in result
    assert "vision" in result
    assert "rag" in result
    assert result["verifiers"]["make"] == "HONDA"
    assert result["risk"]["risk_score"] == pytest.approx(0.12)


def test_external_failure_still_completes(client, policy_pdf, estimate_pdf, monkeypatch):
    def failing_verifiers(*_args, **_kwargs):
        return VerifierOutput(
            sources_failed=[SOURCE_NHTSA_VIN],
            weather_at_incident=None,
        )

    monkeypatch.setattr("worker.tasks.run_verifiers", failing_verifiers)
    monkeypatch.setattr(
        "worker.tasks.run_fraud_agent",
        lambda *_a, **_k: RiskOutput(flags=[], risk_score=0.0),
    )

    files = {
        "policy_pdf": ("sample_policy.pdf", policy_pdf.read_bytes(), "application/pdf"),
        "estimate_pdf": ("sample_estimate.pdf", estimate_pdf.read_bytes(), "application/pdf"),
    }
    create = client.post("/claims", files=files, data={"incident_location": "Boston, MA"})
    claim_id = create.json()["claim_id"]
    payload = client.get(f"/claims/{claim_id}").json()
    assert payload["status"] == "completed"
    assert SOURCE_NHTSA_VIN in payload["result"]["verifiers"]["sources_failed"]
    assert "risk" in payload["result"]
