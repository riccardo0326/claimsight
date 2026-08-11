"""Offline unit tests for External Verifiers (mocked HTTP)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from agents.schemas import DocumentOutput
from agents.verifiers import (
    SOURCE_GEOCODING,
    SOURCE_NHTSA_VIN,
    SOURCE_WEATHER,
    run_verifiers,
)
from api.config import get_settings
from db.models import ExternalApiCache
from db.session import configure_engine, init_db


VIN = "1HGCM82633A004352"


@pytest.fixture
def sqlite_db(tmp_path, monkeypatch):
    db_path = tmp_path / "verifiers.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DB_URL", url)
    get_settings.cache_clear()
    configure_engine(url)
    init_db()
    from db import session as db_session

    assert db_session.SessionLocal is not None
    db = db_session.SessionLocal()
    try:
        yield db
    finally:
        db.close()
        get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_nominatim_sleep(monkeypatch):
    monkeypatch.setattr("agents.verifiers._respect_nominatim_rate_limit", lambda: None)


def _doc(**overrides: Any) -> DocumentOutput:
    base = dict(
        policy_id="POL-1",
        vin=VIN,
        incident_date=date(2024, 3, 14),
        coverage_limits={},
        deductible=500.0,
        line_items=[],
    )
    base.update(overrides)
    return DocumentOutput(**base)


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _RoutingClient:
    """Minimal httpx.Client stand-in that routes by URL substring."""

    def __init__(self, routes: dict[str, Any], *, fail_times: dict[str, int] | None = None):
        self.routes = routes
        self.fail_times = fail_times or {}
        self.calls: list[str] = []
        self._fail_counts: dict[str, int] = {}

    def get(self, url: str, headers=None, params=None, timeout=None):  # noqa: ANN001
        self.calls.append(url)
        # Longer / more specific keys first to avoid substring collisions.
        for key, payload in sorted(self.routes.items(), key=lambda kv: -len(kv[0])):
            if key in url:
                fails_left = self.fail_times.get(key, 0) - self._fail_counts.get(key, 0)
                if fails_left > 0:
                    self._fail_counts[key] = self._fail_counts.get(key, 0) + 1
                    raise httpx.TimeoutException("simulated timeout")
                if isinstance(payload, Exception):
                    raise payload
                if callable(payload):
                    return payload(url, params)
                status = 200
                body = payload
                if isinstance(payload, tuple):
                    status, body = payload
                return _FakeResponse(body, status_code=status)
        raise AssertionError(f"Unexpected URL: {url}")

    def close(self) -> None:
        return None


def _happy_routes() -> dict[str, Any]:
    return {
        "DecodeVinValues": {
            "Results": [
                {
                    "Make": "HONDA",
                    "Model": "Accord",
                    "ModelYear": "2003",
                    "ErrorCode": "0",
                }
            ]
        },
        "recallsByVehicle": {
            "results": [
                {
                    "NHTSACampaignNumber": "03V123000",
                    "Component": "AIR BAGS",
                    "Summary": "Airbag inflator may rupture.",
                }
            ]
        },
        "complaintsByVehicle": {
            "results": [
                {
                    "odiNumber": "1001",
                    "component": "ELECTRICAL SYSTEM",
                    "summary": "Battery drain complaint.",
                }
            ]
        },
        "nominatim": [{"lat": "38.9072", "lon": "-77.0369"}],
        "api.weather.gov/points/": {
            "properties": {
                "observationStations": "https://api.weather.gov/gridpoints/LWX/97,71/stations"
            }
        },
        "gridpoints/LWX/97,71/stations": {
            "features": [
                {
                    "properties": {"stationIdentifier": "KDCA"},
                    "id": "https://api.weather.gov/stations/KDCA",
                }
            ]
        },
        "stations/KDCA/observations": {
            "features": [
                {
                    "properties": {
                        "textDescription": "Thunderstorm",
                        "precipitationLast6Hours": {
                            "value": 0.012,
                            "unitCode": "wmoUnit:m",
                        },
                        "presentWeather": [],
                    }
                }
            ]
        },
    }


def test_happy_path_complete_verifier_output(sqlite_db: Session):
    client = _RoutingClient(_happy_routes())
    out = run_verifiers(
        _doc(),
        incident_location="Washington, DC",
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    assert out.make == "HONDA"
    assert out.model == "Accord"
    assert out.model_year == 2003
    assert len(out.nhtsa_recalls) == 1
    assert out.nhtsa_recalls[0].campaign_number == "03V123000"
    assert len(out.nhtsa_complaints) == 1
    assert out.weather_at_incident is not None
    assert out.weather_at_incident.had_storm_event is True
    assert out.sources_failed == []


def test_vin_decode_failure_skips_recalls_complaints(sqlite_db: Session, monkeypatch):
    monkeypatch.setenv("EXTERNAL_API_MAX_ATTEMPTS", "1")
    get_settings.cache_clear()
    client = _RoutingClient(
        {
            "DecodeVinValues": httpx.TimeoutException("boom"),
            "nominatim": [{"lat": "1.0", "lon": "2.0"}],
        }
    )
    out = run_verifiers(
        _doc(),
        incident_location=None,
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    assert out.make is None
    assert out.nhtsa_recalls == []
    assert out.nhtsa_complaints == []
    assert SOURCE_NHTSA_VIN in out.sources_failed
    assert not any("recalls" in c for c in client.calls)
    assert not any("complaints" in c for c in client.calls)
    get_settings.cache_clear()


def test_weather_timeout_degrades(sqlite_db: Session, monkeypatch):
    monkeypatch.setenv("EXTERNAL_API_MAX_ATTEMPTS", "1")
    get_settings.cache_clear()
    routes = _happy_routes()
    routes["stations/KDCA/observations"] = httpx.TimeoutException("weather timeout")
    client = _RoutingClient(routes)
    out = run_verifiers(
        _doc(),
        incident_location="Washington, DC",
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    assert out.weather_at_incident is None
    assert SOURCE_WEATHER in out.sources_failed
    assert out.make == "HONDA"
    get_settings.cache_clear()


def test_missing_incident_location_skips_geo_weather(sqlite_db: Session):
    client = _RoutingClient(_happy_routes())
    out = run_verifiers(
        _doc(),
        incident_location=None,
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    assert out.weather_at_incident is None
    assert not any("nominatim" in c for c in client.calls)
    assert not any("api.weather.gov" in c for c in client.calls)
    assert SOURCE_GEOCODING not in out.sources_failed
    assert SOURCE_WEATHER not in out.sources_failed


def test_missing_vin_skips_nhtsa(sqlite_db: Session):
    client = _RoutingClient(_happy_routes())
    out = run_verifiers(
        _doc(vin=None),
        incident_location=None,
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    assert out.make is None and out.model is None and out.model_year is None
    assert not any("nhtsa" in c.lower() or "DecodeVin" in c for c in client.calls)


def test_cache_hit_skips_second_http(sqlite_db: Session):
    client = _RoutingClient(_happy_routes())
    run_verifiers(
        _doc(),
        incident_location="Washington, DC",
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    first_calls = list(client.calls)
    assert first_calls
    client.calls.clear()
    run_verifiers(
        _doc(),
        incident_location="Washington, DC",
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    assert client.calls == []


def test_vin_cache_expiry_refetches(sqlite_db: Session):
    client = _RoutingClient(_happy_routes())
    run_verifiers(
        _doc(),
        incident_location=None,
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    # Expire all cache rows.
    rows = sqlite_db.execute(
        __import__("sqlalchemy").select(ExternalApiCache)
    ).scalars().all()
    for row in rows:
        row.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    sqlite_db.commit()
    client.calls.clear()
    run_verifiers(
        _doc(),
        incident_location=None,
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    assert any("DecodeVinValues" in c for c in client.calls)


def test_weather_cache_no_second_http(sqlite_db: Session):
    client = _RoutingClient(_happy_routes())
    run_verifiers(
        _doc(),
        incident_location="Washington, DC",
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    client.calls.clear()
    run_verifiers(
        _doc(),
        incident_location="Washington, DC",
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    assert client.calls == []
    from sqlalchemy import select

    rows = (
        sqlite_db.execute(
            select(ExternalApiCache).where(ExternalApiCache.source == SOURCE_WEATHER)
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1
    assert rows[0].expires_at is None


def test_retry_then_success(sqlite_db: Session, monkeypatch):
    monkeypatch.setenv("EXTERNAL_API_MAX_ATTEMPTS", "2")
    get_settings.cache_clear()
    client = _RoutingClient(
        _happy_routes(),
        fail_times={"DecodeVinValues": 1},
    )
    out = run_verifiers(
        _doc(),
        incident_location=None,
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    assert out.make == "HONDA"
    assert sum(1 for c in client.calls if "DecodeVinValues" in c) == 2
    get_settings.cache_clear()


def test_permanent_failure_max_attempts(sqlite_db: Session, monkeypatch):
    monkeypatch.setenv("EXTERNAL_API_MAX_ATTEMPTS", "2")
    get_settings.cache_clear()
    client = _RoutingClient(
        {"DecodeVinValues": httpx.TimeoutException("always")},
    )
    # Wrap so every call times out (fail_times infinite via Exception payload).
    out = run_verifiers(
        _doc(),
        incident_location=None,
        db=sqlite_db,
        client=client,  # type: ignore[arg-type]
    )
    assert SOURCE_NHTSA_VIN in out.sources_failed
    assert sum(1 for c in client.calls if "DecodeVinValues" in c) == 2
    get_settings.cache_clear()
