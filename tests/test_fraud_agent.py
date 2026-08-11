"""Offline unit tests for Fraud/Risk Agent (mocked zero-shot)."""

from __future__ import annotations

from datetime import date

from agents.fraud_agent import compute_risk_score, run_fraud_agent
from agents.schemas import (
    DocumentOutput,
    LineItem,
    NHTSARecall,
    RiskFlag,
    VerifierOutput,
    WeatherAtIncident,
)


def _doc(**kwargs):
    base = dict(
        policy_id="POL-1",
        vin="1HGCM82633A004352",
        incident_date=date(2024, 3, 14),
        coverage_limits={},
        deductible=500.0,
        line_items=[LineItem(description="Front bumper replacement", cost=850.0)],
    )
    base.update(kwargs)
    return DocumentOutput(**base)


def test_weather_mismatch_rule():
    verifiers = VerifierOutput(
        weather_at_incident=WeatherAtIncident(
            condition="Clear",
            precipitation_mm=0.0,
            had_storm_event=False,
        )
    )
    out = run_fraud_agent(
        "Hail storm damaged the roof and hood.",
        _doc(),
        verifiers,
        classifier_scores={},
    )
    types = {f.flag_type for f in out.flags}
    assert "weather mismatch" in types
    assert 0.0 <= out.risk_score <= 1.0
    assert out.risk_score >= 0.3


def test_recall_related_rule():
    verifiers = VerifierOutput(
        nhtsa_recalls=[
            NHTSARecall(
                campaign_number="99V001000",
                component="BUMPERS",
                summary="Bumper attachment bolts may loosen.",
            )
        ]
    )
    out = run_fraud_agent(
        "Front-end collision damaged the bumper.",
        _doc(),
        verifiers,
        classifier_scores={},
    )
    types = {f.flag_type for f in out.flags}
    assert "recall-related damage" in types
    flag = next(f for f in out.flags if f.flag_type == "recall-related damage")
    assert flag.severity == "info"


def test_empty_narrative_graceful():
    out = run_fraud_agent(
        "",
        _doc(),
        VerifierOutput(),
        classifier_scores={},
    )
    assert out.flags == []
    assert out.risk_score == 0.0


def test_risk_score_bounded():
    flags = [
        RiskFlag(flag_type="possible staged damage", rationale="x", severity="high"),
        RiskFlag(flag_type="inconsistent claim", rationale="y", severity="medium"),
        RiskFlag(flag_type="weather mismatch", rationale="z", severity="medium"),
        RiskFlag(flag_type="recall-related damage", rationale="w", severity="info"),
    ]
    score = compute_risk_score(
        flags=flags,
        classifier_scores={
            "possible staged damage": 0.99,
            "inconsistent claim": 0.99,
            "weather mismatch": 0.99,
            "recall-related damage": 0.99,
        },
    )
    assert 0.0 <= score <= 1.0


def test_zero_shot_invocation_mocked(monkeypatch):
    calls: list[str] = []

    def fake_classify(narrative: str):
        calls.append(narrative)
        return {"inconsistent claim": 0.8, "consistent claim": 0.1}

    monkeypatch.setattr("agents.fraud_agent._classify_narrative", fake_classify)
    out = run_fraud_agent(
        "Something odd about this claim narrative.",
        _doc(),
        VerifierOutput(),
    )
    assert calls == ["Something odd about this claim narrative."]
    assert any(f.flag_type == "inconsistent claim" for f in out.flags)
    assert 0.0 <= out.risk_score <= 1.0
