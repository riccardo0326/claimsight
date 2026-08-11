"""External Verifiers — NHTSA + Nominatim + NOAA/NWS (Slice 4).

Contract: VerifierOutput in agents/schemas.py. Cache + retry policy in DECISIONS.md.
Failed sources degrade into sources_failed; they never fail the claim.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agents.schemas import (
    DocumentOutput,
    NHTSAComplaint,
    NHTSARecall,
    VerifierOutput,
    WeatherAtIncident,
)
from api.config import get_settings
from db.models import ExternalApiCache

logger = logging.getLogger(__name__)

NHTSA_VIN_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}"
NHTSA_RECALLS_URL = "https://api.nhtsa.gov/recalls/recallsByVehicle"
NHTSA_COMPLAINTS_URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"

SOURCE_NHTSA_VIN = "nhtsa_vin"
SOURCE_NHTSA_RECALLS = "nhtsa_recalls"
SOURCE_NHTSA_COMPLAINTS = "nhtsa_complaints"
SOURCE_GEOCODING = "geocoding"
SOURCE_WEATHER = "weather"

# Cap list sizes persisted on claims (NHTSA can return hundreds of complaints).
MAX_NHTSA_RECALLS = 50
MAX_NHTSA_COMPLAINTS = 50

# Nominatim usage policy: max 1 request/second.
_nominatim_lock = threading.Lock()
_nominatim_last_request_monotonic: float = 0.0


class ExternalApiError(Exception):
    """Raised when an external HTTP call fails after retries."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }


def _cache_get(db: Session, cache_key: str) -> dict | None:
    row = db.execute(
        select(ExternalApiCache).where(ExternalApiCache.cache_key == cache_key)
    ).scalar_one_or_none()
    if row is None:
        return None
    if row.expires_at is not None:
        expires = row.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= _utcnow():
            return None
    return row.response_json


def _cache_put(
    db: Session,
    *,
    cache_key: str,
    source: str,
    response_json: dict,
    expires_at: datetime | None,
) -> None:
    existing = db.execute(
        select(ExternalApiCache).where(ExternalApiCache.cache_key == cache_key)
    ).scalar_one_or_none()
    now = _utcnow()
    if existing is None:
        db.add(
            ExternalApiCache(
                cache_key=cache_key,
                source=source,
                response_json=response_json,
                fetched_at=now,
                expires_at=expires_at,
            )
        )
    else:
        existing.source = source
        existing.response_json = response_json
        existing.fetched_at = now
        existing.expires_at = expires_at
    db.commit()


def _get_json(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    timeout: float,
    max_attempts: int,
) -> Any:
    """GET JSON with Tenacity retry (max attempts, exponential backoff)."""

    @retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, ExternalApiError)),
    )
    def _once() -> Any:
        try:
            resp = client.get(url, headers=headers, params=params, timeout=timeout)
        except (httpx.TimeoutException, httpx.TransportError):
            raise
        if resp.status_code >= 500:
            raise ExternalApiError(f"HTTP {resp.status_code} for {url}")
        if resp.status_code >= 400:
            raise ExternalApiError(f"HTTP {resp.status_code} for {url}")
        try:
            return resp.json()
        except ValueError as exc:
            raise ExternalApiError(f"Invalid JSON from {url}") from exc

    return _once()


def _respect_nominatim_rate_limit() -> None:
    global _nominatim_last_request_monotonic
    with _nominatim_lock:
        now = time.monotonic()
        wait_for = 1.0 - (now - _nominatim_last_request_monotonic)
        if wait_for > 0:
            time.sleep(wait_for)
        _nominatim_last_request_monotonic = time.monotonic()


def _nhtsa_vin_cache_key(vin: str) -> str:
    return f"nhtsa_vin:{vin.upper()}"


def _nhtsa_vehicle_cache_key(source: str, make: str, model: str, year: int) -> str:
    return f"{source}:{make.lower()}:{model.lower()}:{year}"


def _geocode_cache_key(location: str) -> str:
    return f"geocoding:{location.strip().lower()}"


def _weather_cache_key(lat: float, lon: float, incident_date: date) -> str:
    return f"weather:{lat:.4f}:{lon:.4f}:{incident_date.isoformat()}"


def _parse_vin_decode(payload: dict) -> tuple[str | None, str | None, int | None]:
    results = payload.get("Results") or []
    if not results:
        return None, None, None
    row = results[0] if isinstance(results[0], dict) else {}
    make = (row.get("Make") or "").strip() or None
    model = (row.get("Model") or "").strip() or None
    year_raw = (row.get("ModelYear") or "").strip()
    year: int | None = None
    if year_raw:
        try:
            year = int(year_raw)
        except ValueError:
            year = None
    # ErrorCode "0" means successful decode; non-zero / missing identity → treat as fail.
    error = str(row.get("ErrorCode") or "")
    if make is None or model is None or year is None:
        return None, None, None
    if error and not error.startswith("0"):
        # Some responses use "0" or "0, ..." — reject clearly failed codes.
        primary = error.split(",")[0].strip()
        if primary not in {"0", ""}:
            return None, None, None
    return make, model, year


def _parse_recalls(payload: dict) -> list[NHTSARecall]:
    out: list[NHTSARecall] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        campaign = str(
            item.get("NHTSACampaignNumber") or item.get("CampaignNumber") or ""
        ).strip()
        component = str(item.get("Component") or "").strip()
        summary = str(item.get("Summary") or "").strip()
        if not campaign:
            continue
        out.append(
            NHTSARecall(
                campaign_number=campaign,
                component=component or "unknown",
                summary=summary or "",
            )
        )
    return out


def _parse_complaints(payload: dict) -> list[NHTSAComplaint]:
    out: list[NHTSAComplaint] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        cid = str(
            item.get("odiNumber")
            or item.get("ODINumber")
            or item.get("ComplaintId")
            or item.get("id")
            or ""
        ).strip()
        component = str(item.get("component") or item.get("Component") or "").strip()
        summary = str(
            item.get("summary")
            or item.get("Summary")
            or item.get("complaint")
            or ""
        ).strip()
        if not cid:
            continue
        out.append(
            NHTSAComplaint(
                complaint_id=cid,
                component=component or "unknown",
                summary=summary or "",
            )
        )
    return out


def _parse_geocode(payload: Any) -> tuple[float, float] | None:
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    if not isinstance(first, dict):
        return None
    try:
        return float(first["lat"]), float(first["lon"])
    except (KeyError, TypeError, ValueError):
        return None


def _qty_to_mm(value: Any) -> float | None:
    """Convert NWS QuantitativeValue precipitation to millimeters."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # NWS precip values are typically meters when unitCode is wmoUnit:m.
        return float(value) * 1000.0
    if isinstance(value, dict):
        raw = value.get("value")
        if raw is None:
            return None
        unit = str(value.get("unitCode") or value.get("uom") or "").lower()
        try:
            num = float(raw)
        except (TypeError, ValueError):
            return None
        if "mm" in unit:
            return num
        if unit.endswith(":m") or unit.endswith("/m") or unit == "m":
            return num * 1000.0
        if "in" in unit:  # inches
            return num * 25.4
        # Default NWS observation precip: meters.
        return num * 1000.0
    return None


def _map_weather_from_observations(
    observations: list[dict],
    *,
    storm_precip_mm: float,
) -> WeatherAtIncident | None:
    """Map NWS station observations for the incident day into WeatherAtIncident.

    Mapping (documented in DECISIONS.md):
    - condition: first non-empty textDescription among observations
    - precipitation_mm: max of precipitationLast* QuantitativeValues converted to mm
    - had_storm_event: True if precip >= storm threshold OR text/weather codes
      mention thunder/storm/hail/tornado/severe
    """
    if not observations:
        return None

    condition = "unknown"
    precip_values: list[float] = []
    storm_text = False
    storm_tokens = ("thunder", "storm", "hail", "tornado", "severe")

    for obs in observations:
        props = obs.get("properties") if isinstance(obs, dict) else None
        if not isinstance(props, dict):
            continue
        text = str(props.get("textDescription") or "").strip()
        if text and condition == "unknown":
            condition = text
        lower = text.lower()
        if any(tok in lower for tok in storm_tokens):
            storm_text = True
        for key in (
            "precipitationLastHour",
            "precipitationLast3Hours",
            "precipitationLast6Hours",
        ):
            mm = _qty_to_mm(props.get(key))
            if mm is not None:
                precip_values.append(mm)
        present = props.get("presentWeather") or []
        if isinstance(present, list):
            for entry in present:
                blob = str(entry).lower()
                if any(tok in blob for tok in storm_tokens):
                    storm_text = True

    precip_mm = max(precip_values) if precip_values else 0.0
    had_storm = storm_text or precip_mm >= storm_precip_mm
    return WeatherAtIncident(
        condition=condition,
        precipitation_mm=round(precip_mm, 2),
        had_storm_event=had_storm,
    )


def run_verifiers(
    document: DocumentOutput,
    *,
    incident_location: str | None,
    db: Session,
    client: httpx.Client | None = None,
) -> VerifierOutput:
    """Run External Verifiers with cache + graceful degradation."""
    settings = get_settings()
    timeout = settings.external_api_timeout_seconds
    max_attempts = settings.external_api_max_attempts
    headers = _headers(settings.http_user_agent)
    ttl = timedelta(hours=settings.nhtsa_cache_ttl_hours)

    sources_failed: list[str] = []
    make: str | None = None
    model: str | None = None
    model_year: int | None = None
    recalls: list[NHTSARecall] = []
    complaints: list[NHTSAComplaint] = []
    weather: WeatherAtIncident | None = None

    owns_client = client is None
    http = client or httpx.Client()
    try:
        vin = (document.vin or "").strip() or None
        if vin:
            vin_key = _nhtsa_vin_cache_key(vin)
            cached = _cache_get(db, vin_key)
            try:
                if cached is not None:
                    payload = cached
                else:
                    url = NHTSA_VIN_URL.format(vin=quote(vin, safe=""))
                    payload = _get_json(
                        http,
                        url,
                        headers=headers,
                        params={"format": "json"},
                        timeout=timeout,
                        max_attempts=max_attempts,
                    )
                    if not isinstance(payload, dict):
                        raise ExternalApiError("VIN decode payload is not an object")
                    _cache_put(
                        db,
                        cache_key=vin_key,
                        source=SOURCE_NHTSA_VIN,
                        response_json=payload,
                        expires_at=_utcnow() + ttl,
                    )
                make, model, model_year = _parse_vin_decode(payload)
                if make is None or model is None or model_year is None:
                    sources_failed.append(SOURCE_NHTSA_VIN)
                    logger.info("NHTSA VIN decode unusable for vin=%s", vin)
            except Exception:  # noqa: BLE001 — degrade
                logger.exception("NHTSA VIN decode failed")
                sources_failed.append(SOURCE_NHTSA_VIN)

            if make and model and model_year is not None:
                # Recalls
                recall_key = _nhtsa_vehicle_cache_key(
                    SOURCE_NHTSA_RECALLS, make, model, model_year
                )
                try:
                    cached_r = _cache_get(db, recall_key)
                    if cached_r is not None:
                        recall_payload = cached_r
                    else:
                        recall_payload = _get_json(
                            http,
                            NHTSA_RECALLS_URL,
                            headers=headers,
                            params={
                                "make": make,
                                "model": model,
                                "modelYear": model_year,
                            },
                            timeout=timeout,
                            max_attempts=max_attempts,
                        )
                        if not isinstance(recall_payload, dict):
                            raise ExternalApiError("recalls payload is not an object")
                        _cache_put(
                            db,
                            cache_key=recall_key,
                            source=SOURCE_NHTSA_RECALLS,
                            response_json=recall_payload,
                            expires_at=_utcnow() + ttl,
                        )
                    recalls = _parse_recalls(recall_payload)[:MAX_NHTSA_RECALLS]
                except Exception:  # noqa: BLE001
                    logger.exception("NHTSA recalls failed")
                    sources_failed.append(SOURCE_NHTSA_RECALLS)

                # Complaints
                complaint_key = _nhtsa_vehicle_cache_key(
                    SOURCE_NHTSA_COMPLAINTS, make, model, model_year
                )
                try:
                    cached_c = _cache_get(db, complaint_key)
                    if cached_c is not None:
                        complaint_payload = cached_c
                    else:
                        complaint_payload = _get_json(
                            http,
                            NHTSA_COMPLAINTS_URL,
                            headers=headers,
                            params={
                                "make": make,
                                "model": model,
                                "modelYear": model_year,
                            },
                            timeout=timeout,
                            max_attempts=max_attempts,
                        )
                        if not isinstance(complaint_payload, dict):
                            raise ExternalApiError("complaints payload is not an object")
                        _cache_put(
                            db,
                            cache_key=complaint_key,
                            source=SOURCE_NHTSA_COMPLAINTS,
                            response_json=complaint_payload,
                            expires_at=_utcnow() + ttl,
                        )
                    complaints = _parse_complaints(complaint_payload)[:MAX_NHTSA_COMPLAINTS]
                except Exception:  # noqa: BLE001
                    logger.exception("NHTSA complaints failed")
                    sources_failed.append(SOURCE_NHTSA_COMPLAINTS)

        # Geocode + weather (both require location; weather also needs date).
        location = (incident_location or "").strip() or None
        incident_date = document.incident_date
        coords: tuple[float, float] | None = None

        if location is None:
            logger.info("Skipping geocoding/weather: incident_location missing")
        else:
            geo_key = _geocode_cache_key(location)
            try:
                cached_g = _cache_get(db, geo_key)
                if cached_g is not None:
                    coords = (float(cached_g["lat"]), float(cached_g["lon"]))
                else:
                    _respect_nominatim_rate_limit()
                    geo_list = _get_json(
                        http,
                        NOMINATIM_URL,
                        headers={
                            **headers,
                            "Accept-Language": "en",
                        },
                        params={
                            "q": location,
                            "format": "json",
                            "limit": 1,
                            # Nominatim asks for a contact email in the query when possible.
                            "email": "claimsight-dev@example.com",
                        },
                        timeout=timeout,
                        max_attempts=max_attempts,
                    )
                    coords = _parse_geocode(geo_list)
                    if coords is None:
                        raise ExternalApiError("geocoding returned no coordinates")
                    _cache_put(
                        db,
                        cache_key=geo_key,
                        source=SOURCE_GEOCODING,
                        response_json={
                            "lat": coords[0],
                            "lon": coords[1],
                            "results": geo_list,
                        },
                        expires_at=_utcnow() + ttl,
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Geocoding failed")
                sources_failed.append(SOURCE_GEOCODING)
                coords = None

        if location is None or incident_date is None:
            if location is not None and incident_date is None:
                logger.info("Skipping weather: incident_date missing")
                # Location present but no date — weather cannot run; not a hard API failure.
            weather = None
        elif coords is None:
            weather = None
            # Geocoding already recorded failure when applicable.
        else:
            lat, lon = coords
            weather_key = _weather_cache_key(lat, lon, incident_date)
            try:
                cached_w = _cache_get(db, weather_key)
                if cached_w is not None and "weather" in cached_w:
                    weather = WeatherAtIncident.model_validate(cached_w["weather"])
                else:
                    points_url = NWS_POINTS_URL.format(lat=f"{lat:.4f}", lon=f"{lon:.4f}")
                    points = _get_json(
                        http,
                        points_url,
                        headers={**headers, "Accept": "application/geo+json"},
                        timeout=timeout,
                        max_attempts=max_attempts,
                    )
                    if not isinstance(points, dict):
                        raise ExternalApiError("NWS points payload invalid")
                    stations_url = (
                        (points.get("properties") or {}).get("observationStations")
                    )
                    if not stations_url:
                        raise ExternalApiError("NWS points missing observationStations")
                    stations_payload = _get_json(
                        http,
                        stations_url,
                        headers={**headers, "Accept": "application/geo+json"},
                        timeout=timeout,
                        max_attempts=max_attempts,
                    )
                    features = (stations_payload or {}).get("features") or []
                    if not features:
                        raise ExternalApiError("No NWS observation stations nearby")
                    station_id = (
                        (features[0].get("properties") or {}).get("stationIdentifier")
                        or (features[0].get("id") or "").rstrip("/").split("/")[-1]
                    )
                    if not station_id:
                        raise ExternalApiError("NWS station id missing")
                    start = f"{incident_date.isoformat()}T00:00:00Z"
                    end = f"{incident_date.isoformat()}T23:59:59Z"
                    obs_url = f"https://api.weather.gov/stations/{station_id}/observations"
                    obs_payload = _get_json(
                        http,
                        obs_url,
                        headers={**headers, "Accept": "application/geo+json"},
                        params={"start": start, "end": end},
                        timeout=timeout,
                        max_attempts=max_attempts,
                    )
                    obs_features = (obs_payload or {}).get("features") or []
                    weather = _map_weather_from_observations(
                        obs_features,
                        storm_precip_mm=settings.weather_storm_precip_mm,
                    )
                    if weather is None:
                        raise ExternalApiError("No usable observations for incident date")
                    _cache_put(
                        db,
                        cache_key=weather_key,
                        source=SOURCE_WEATHER,
                        response_json={
                            "weather": weather.model_dump(mode="json"),
                            "station_id": station_id,
                        },
                        expires_at=None,  # historical weather — no TTL
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Weather lookup failed")
                sources_failed.append(SOURCE_WEATHER)
                weather = None

    finally:
        if owns_client:
            http.close()

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_failed: list[str] = []
    for src in sources_failed:
        if src not in seen:
            seen.add(src)
            unique_failed.append(src)

    return VerifierOutput(
        make=make,
        model=model,
        model_year=model_year,
        nhtsa_recalls=recalls,
        nhtsa_complaints=complaints,
        weather_at_incident=weather,
        sources_failed=unique_failed,
    )
