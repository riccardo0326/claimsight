import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, Integer, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from db.base import Base

# all-MiniLM-L6-v2 embedding dimension
EMBEDDING_DIM = 384


class ClaimStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# Portable JSON that becomes JSONB on Postgres and JSON elsewhere (SQLite tests).
PortableJSON = JSON().with_variant(JSONB(), "postgresql")


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    status: Mapped[ClaimStatus] = mapped_column(
        Enum(ClaimStatus, name="claim_status", native_enum=False),
        nullable=False,
        default=ClaimStatus.pending,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    input_paths: Mapped[dict] = mapped_column(PortableJSON, nullable=False)
    # Optional for Slice 1 curl compatibility; real claims should require a narrative.
    narrative: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    # Optional incident location for External Verifiers (Slice 4). Missing → skip weather.
    incident_location: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    result: Mapped[dict | None] = mapped_column(PortableJSON, nullable=True)


class PolicyClause(Base):
    """Postgres+pgvector only — not created on SQLite (see db.session.init_db)."""

    __tablename__ = "policy_clauses"
    __table_args__ = (
        UniqueConstraint("policy_id", "clause_id", name="uq_policy_clause"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    policy_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    clause_id: Mapped[str] = mapped_column(String(128), nullable=False)
    clause_type: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class ExternalApiCache(Base):
    """Cached external verifier responses (SQLite + Postgres). See DECISIONS.md Slice 4."""

    __tablename__ = "external_api_cache"
    __table_args__ = (UniqueConstraint("cache_key", name="uq_external_api_cache_key"),)

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    cache_key: Mapped[str] = mapped_column(String(512), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    response_json: Mapped[dict] = mapped_column(PortableJSON, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    # Null expires_at = never expires (used for historical weather).
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
