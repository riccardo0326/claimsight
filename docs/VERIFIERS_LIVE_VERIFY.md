# Slice 4 — External Verifiers live verification

Manual procedure (network required). Not part of default `pytest`.

## Prerequisites

- Python 3.12 venv with project deps (`pip install -e ".[dev]"` / `requirements.txt`)
- Outbound HTTPS to:
  - `vpic.nhtsa.dot.gov`
  - `api.nhtsa.gov`
  - `nominatim.openstreetmap.org`
  - `api.weather.gov`

No API keys. Nominatim requires a descriptive User-Agent (configured via
`HTTP_USER_AGENT` / Settings default).

## Run

```bash
python scripts/verify_verifiers_live.py

# Optional pytest marker (excluded from default suite)
pytest -m live_api
```

Default live inputs:

| Input | Value |
|-------|--------|
| VIN | `1HGCM82633A004352` |
| Location | `Washington, DC` |
| Incident date | `2024-03-14` |

## Checklist

1. VIN resolves to make / model / model_year
2. Recalls and complaints are queried with that identity (not raw VIN)
3. Nominatim returns coordinates
4. NWS points → station observations map into `WeatherAtIncident`
5. Second identical NHTSA call is served from `external_api_cache`
6. Missing `incident_date` skips weather without failing
7. Fraud/Risk still produces a `RiskOutput` (uses synthetic clear weather if live weather unavailable)

## Results log (2026-08-11)

| Service | Result |
|---------|--------|
| VIN decode | verified against real service → HONDA Accord 2003 |
| NHTSA recalls | verified against real service (24 mapped, capped) |
| NHTSA complaints | verified against real service (capped at 50) |
| Nominatim | failed — public instance returned HTTP 403 (`Access denied` per OSM Nominatim policy; common for some egress IPs). Offline unit tests cover geocoding with mocked HTTP. |
| NOAA/NWS weather | not verified live — blocked on geocoding; offline unit tests cover points→stations→observations mapping and timeout degrade |
| Cache second call | verified (NHTSA VIN/recalls/complaints served from `external_api_cache`) |
| Degrade path | verified (missing date → `weather_at_incident=null`) |
| Fraud/Risk | verified with synthetic clear-weather verifier output → `weather mismatch` flag |

Re-run after moving to a network that Nominatim allows (or a self-hosted Nominatim) to complete weather live checks.
