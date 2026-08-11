"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from api.config import get_settings
from api.routes import claims
from db.session import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title="ClaimSight",
    description="Insurance claims triage — multimodal agents through Fraud/Risk (Slice 4)",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(claims.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
