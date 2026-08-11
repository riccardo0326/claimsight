"""Manual / live verification for Slice 4 External Verifiers.

Not part of the default pytest suite. Prefer:

  pytest -m live_api
  # or
  python scripts/verify_verifiers_live.py

Requires network access. Uses a well-known public VIN and a historical US
location/date. No API keys required.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.fraud_agent import run_fraud_agent
from agents.schemas import DocumentOutput, VerifierOutput, WeatherAtIncident
from agents.verifiers import SOURCE_NHTSA_VIN, run_verifiers
from api.config import get_settings
from db.models import ExternalApiCache
from db.session import configure_engine, init_db

# Public sample VIN commonly used in NHTSA demos (Honda Accord).
LIVE_VIN = "1HGCM82633A004352"
LIVE_LOCATION = "Washington, DC"
LIVE_DATE = date(2024, 3, 14)


def _session(tmp_db: Path):
    url = f"sqlite:///{tmp_db.as_posix()}"
    get_settings.cache_clear()
    configure_engine(url)
    init_db()
    from db import session as db_session

    assert db_session.SessionLocal is not None
    return db_session.SessionLocal()


def _summarize(out: VerifierOutput) -> dict:
    return {
        "make": out.make,
        "model": out.model,
        "model_year": out.model_year,
        "nhtsa_recalls": len(out.nhtsa_recalls),
        "nhtsa_complaints": len(out.nhtsa_complaints),
        "weather_at_incident": (
            out.weather_at_incident.model_dump(mode="json")
            if out.weather_at_incident
            else None
        ),
        "sources_failed": out.sources_failed,
    }


@pytest.mark.live_api
def test_live_nhtsa_nominatim_nws(tmp_path):
    db = _session(tmp_path / "live_verifiers.db")
    try:
        doc = DocumentOutput(vin=LIVE_VIN, incident_date=LIVE_DATE)
        out = run_verifiers(doc, incident_location=LIVE_LOCATION, db=db)
        assert out.make
        assert out.model
        assert out.model_year
        print(json.dumps(_summarize(out), indent=2))
    finally:
        db.close()
        get_settings.cache_clear()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("storage/live_verifiers_cache.db"),
        help="SQLite path for external_api_cache during live run",
    )
    args = parser.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    # Fresh DB each run so cache behavior is measurable.
    if args.db.exists():
        args.db.unlink()

    db = _session(args.db)
    report: dict[str, str] = {}
    try:
        print("=== Live External Verifiers ===")
        print(f"VIN={LIVE_VIN} location={LIVE_LOCATION!r} date={LIVE_DATE}")
        doc = DocumentOutput(vin=LIVE_VIN, incident_date=LIVE_DATE)

        with httpx.Client() as client:
            first = run_verifiers(
                doc, incident_location=LIVE_LOCATION, db=db, client=client
            )
        print("First call summary:", json.dumps(_summarize(first), indent=2))

        report["VIN decode"] = (
            "verified against real service"
            if first.make and SOURCE_NHTSA_VIN not in first.sources_failed
            else "failed"
        )
        report["NHTSA recalls"] = (
            "verified against real service"
            if "nhtsa_recalls" not in first.sources_failed
            else "failed"
        )
        report["NHTSA complaints"] = (
            "verified against real service"
            if "nhtsa_complaints" not in first.sources_failed
            else "failed"
        )
        report["Nominatim"] = (
            "verified against real service"
            if "geocoding" not in first.sources_failed
            else "failed"
        )
        report["NOAA/NWS weather"] = (
            "verified against real service"
            if first.weather_at_incident is not None
            else (
                "failed"
                if "weather" in first.sources_failed or "geocoding" in first.sources_failed
                else "not run"
            )
        )

        if not first.make:
            print("LIVE VERIFY FAILED: VIN decode did not return identity", file=sys.stderr)
            return 1

        # Cache hit for VIN-keyed sources: second call must not HTTP for DecodeVinValues.
        class _Guard:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def get(self, url: str, *args, **kwargs):  # noqa: ANN002, ANN003
                self.calls.append(url)
                if "DecodeVinValues" in url or "recallsByVehicle" in url or "complaintsByVehicle" in url:
                    raise AssertionError(f"NHTSA cache miss for {url}")
                # Allow geocode/weather retries if they failed the first time.
                raise httpx.HTTPStatusError(
                    "blocked",
                    request=httpx.Request("GET", url),
                    response=httpx.Response(403),
                )

            def close(self) -> None:
                return None

        guard = _Guard()
        second = run_verifiers(
            doc, incident_location=LIVE_LOCATION, db=db, client=guard  # type: ignore[arg-type]
        )
        assert second.make == first.make
        report["Cache second call"] = "verified"
        print("NHTSA cache-only second call OK")

        rows = db.execute(select(ExternalApiCache)).scalars().all()
        print(f"Cache rows: {len(rows)} -> {[r.source for r in rows]}")

        degraded = run_verifiers(
            DocumentOutput(vin=LIVE_VIN, incident_date=None),
            incident_location=LIVE_LOCATION,
            db=db,
            client=guard,  # type: ignore[arg-type]
        )
        assert degraded.weather_at_incident is None
        report["Degrade path"] = "verified"
        print("Missing-date weather skip OK")

        # Risk path: use synthetic weather if live weather unavailable.
        verifiers_for_risk = first
        if first.weather_at_incident is None:
            verifiers_for_risk = first.model_copy(
                update={
                    "weather_at_incident": WeatherAtIncident(
                        condition="Clear",
                        precipitation_mm=0.0,
                        had_storm_event=False,
                    )
                }
            )
        risk = run_fraud_agent(
            "Hail damaged the hood during a storm.",
            doc,
            verifiers_for_risk,
            classifier_scores={"consistent claim": 0.7},
        )
        print("Risk:", json.dumps(risk.model_dump(mode="json"), indent=2))
        assert any(f.flag_type == "weather mismatch" for f in risk.flags)

        print("\n=== Live verification report ===")
        for k, v in report.items():
            print(f"{k}: {v}")
        print("LIVE VERIFY OK")
        return 0
    except Exception as exc:  # noqa: BLE001
        print("LIVE VERIFY FAILED:", exc, file=sys.stderr)
        for k, v in report.items():
            print(f"{k}: {v}", file=sys.stderr)
        return 1
    finally:
        db.close()
        get_settings.cache_clear()


if __name__ == "__main__":
    raise SystemExit(main())
