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


def init_db() -> None:
    """Create tables if they do not exist (Alembic deferred to a later slice)."""
    from db.base import Base
    from db.models import Claim, PolicyClause

    eng = ensure_engine()
    if eng.dialect.name == "postgresql":
        with eng.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(bind=eng)
    else:
        # policy_clauses requires pgvector — skip on SQLite (Slice 1 tests).
        Claim.__table__.create(bind=eng, checkfirst=True)
        _ = PolicyClause  # keep import used for registration clarity
