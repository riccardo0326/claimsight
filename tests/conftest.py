"""Shared pytest fixtures for ClaimSight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings
from agents.extractors.fake import fake_from_expected
from agents.schemas import RAGOutput
from db.base import Base
from db.session import configure_engine, init_db

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
EXPECTED_PATH = FIXTURES_DIR / "expected.json"
POLICY_PDF = FIXTURES_DIR / "sample_policy.pdf"
ESTIMATE_PDF = FIXTURES_DIR / "sample_estimate.pdf"
SAMPLE_CLAUSES = FIXTURES_DIR / "sample_policy_clauses.json"
OTHER_CLAUSES = FIXTURES_DIR / "other_policy_clauses.json"


@pytest.fixture(scope="session")
def expected() -> dict:
    return json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sample_clauses() -> dict:
    return json.loads(SAMPLE_CLAUSES.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def other_clauses() -> dict:
    return json.loads(OTHER_CLAUSES.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def policy_pdf() -> Path:
    assert POLICY_PDF.exists(), "Run: python fixtures/generate_fixtures.py"
    return POLICY_PDF


@pytest.fixture(scope="session")
def estimate_pdf() -> Path:
    assert ESTIMATE_PDF.exists(), "Run: python fixtures/generate_fixtures.py"
    return ESTIMATE_PDF


@pytest.fixture
def fake_extractor(expected: dict):
    return fake_from_expected(expected)


def _patch_celery_eager(monkeypatch):
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("REDIS_URL", "memory://")
    get_settings.cache_clear()

    from worker import celery_app as celery_module

    celery_module.celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        broker_url="memory://",
        result_backend="cache+memory://",
    )


def _patch_fake_docvqa(monkeypatch, fake_extractor):
    monkeypatch.setattr(
        "agents.document_agent.get_layoutlm_extractor",
        lambda: fake_extractor,
    )
    monkeypatch.setattr(
        "worker.tasks.run_document_agent",
        lambda policy, estimate: __import__(
            "agents.document_agent", fromlist=["run_document_agent"]
        ).run_document_agent(policy, estimate, extractor=fake_extractor),
    )


@pytest.fixture
def client(tmp_path, monkeypatch, fake_extractor):
    """FastAPI TestClient with SQLite + eager Celery + fake DocVQA + stubbed RAG."""
    db_path = tmp_path / "test.db"
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()

    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("STORAGE_DIR", str(storage_dir))
    _patch_celery_eager(monkeypatch)
    configure_engine(f"sqlite:///{db_path.as_posix()}")

    from db import session as db_session

    _patch_fake_docvqa(monkeypatch, fake_extractor)
    # SQLite has no pgvector — stub RAG so Slice 1 paths still complete.
    monkeypatch.setattr(
        "worker.tasks.run_rag_agent",
        lambda **_kwargs: RAGOutput(retrieved_clauses=[]),
    )

    Base.metadata.drop_all(bind=db_session.engine)
    init_db()

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()


@pytest.fixture(scope="session")
def pgvector_url():
    """Start pgvector/pgvector:pg16 via testcontainers (requires Docker)."""
    pytest.importorskip("testcontainers")
    try:
        from testcontainers.community.postgres import PostgresContainer
    except ImportError:  # pragma: no cover — older testcontainers
        from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(
        image="pgvector/pgvector:pg16",
        username="claimsight",
        password="claimsight",
        dbname="claimsight",
        driver="psycopg2",
    )
    container.start()
    try:
        url = container.get_connection_url()
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        yield url
    finally:
        container.stop()


@pytest.fixture
def pg_session(pgvector_url, monkeypatch):
    monkeypatch.setenv("DB_URL", pgvector_url)
    get_settings.cache_clear()
    configure_engine(pgvector_url)
    from sqlalchemy import text

    from db import session as db_session

    assert db_session.engine is not None
    with db_session.engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(bind=db_session.engine)
    init_db()
    assert db_session.SessionLocal is not None
    db = db_session.SessionLocal()
    try:
        yield db
    finally:
        db.close()
        get_settings.cache_clear()


@pytest.fixture
def rag_client(tmp_path, monkeypatch, fake_extractor, pgvector_url, expected):
    """API client on Postgres+pgvector with sample clauses ingested."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()

    monkeypatch.setenv("DB_URL", pgvector_url)
    monkeypatch.setenv("STORAGE_DIR", str(storage_dir))
    _patch_celery_eager(monkeypatch)
    configure_engine(pgvector_url)

    from sqlalchemy import text

    from db import session as db_session
    from rag.ingest import ingest_file

    assert db_session.engine is not None
    with db_session.engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(bind=db_session.engine)
    init_db()
    assert db_session.SessionLocal is not None
    with db_session.SessionLocal() as db:
        ingest_file(SAMPLE_CLAUSES, db=db)

    _patch_fake_docvqa(monkeypatch, fake_extractor)

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()
