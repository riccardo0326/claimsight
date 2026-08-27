from collections.abc import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from api.config import get_settings

engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None


def configure_engine(db_url: str | None = None) -> Engine:
    """(Re)configure the global engine and session factory.

    Called on first use (or explicitly from tests) so importing this module
    does not require the production DB driver to be installed.
    """
    global engine, SessionLocal

    url = db_url if db_url is not None else get_settings().db_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, future=True, connect_args=connect_args)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _connection_record):  # noqa: ANN001
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    return engine


def ensure_engine() -> Engine:
    if engine is None or SessionLocal is None:
        return configure_engine()
    return engine


def get_db() -> Generator[Session, None, None]:
    ensure_engine()
    assert SessionLocal is not None
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_claim_columns(eng: Engine) -> None:
    """Add columns introduced after the initial claims table (no Alembic yet).

    ``create_all`` does not ALTER existing tables, so older Docker volumes miss
    later-slice columns and POST /claims fails with a 500.
    """
    dialect = eng.dialect.name
    alters: list[str]
    if dialect == "postgresql":
        alters = [
            "ALTER TABLE claims ADD COLUMN IF NOT EXISTS narrative TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE claims ADD COLUMN IF NOT EXISTS incident_location TEXT",
        ]
    elif dialect == "sqlite":
        # SQLite has no IF NOT EXISTS for ADD COLUMN — probe first.
        with eng.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(claims)")).fetchall()
        existing = {row[1] for row in rows}  # row[1] = name
        alters = []
        if "narrative" not in existing:
            alters.append(
                "ALTER TABLE claims ADD COLUMN narrative TEXT NOT NULL DEFAULT ''"
            )
        if "incident_location" not in existing:
            alters.append("ALTER TABLE claims ADD COLUMN incident_location TEXT")
    else:
        return

    if not alters:
        return
    with eng.begin() as conn:
        for stmt in alters:
            conn.execute(text(stmt))


def init_db() -> None:
    """Create tables if they do not exist (Alembic deferred to a later slice)."""
    from db.base import Base
    from db.models import Claim, ExternalApiCache, PolicyClause

    eng = ensure_engine()
    if eng.dialect.name == "postgresql":
        with eng.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=eng)
        _ensure_claim_columns(eng)
    else:
        # policy_clauses requires pgvector — skip on SQLite (Slice 1 tests).
        Claim.__table__.create(bind=eng, checkfirst=True)
        ExternalApiCache.__table__.create(bind=eng, checkfirst=True)
        _ensure_claim_columns(eng)
        _ = PolicyClause  # keep import used for registration clarity
