"""Policy clause ingestion: LlamaIndex documents → embed → policy_clauses.

Fixture JSON clauses are already clause-sized, so we do not re-chunk aggressively.
PDF-level chunking is deferred to a later slice when ingesting raw policy PDFs.
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path

from llama_index.core import Document
from sqlalchemy import delete
from sqlalchemy.orm import Session

from db.models import PolicyClause
from db import session as db_session
from rag.embeddings import embed_texts

logger = logging.getLogger(__name__)


def load_clause_file(path: Path) -> tuple[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    policy_id = data["policy_id"]
    clauses = data["clauses"]
    if not isinstance(clauses, list) or not clauses:
        raise ValueError(f"No clauses found in {path}")
    return policy_id, clauses


def clauses_to_documents(policy_id: str, clauses: list[dict]) -> list[Document]:
    docs: list[Document] = []
    for clause in clauses:
        docs.append(
            Document(
                text=clause["text"],
                metadata={
                    "policy_id": policy_id,
                    "clause_id": clause["clause_id"],
                    "clause_type": clause["clause_type"],
                    "page_number": int(clause.get("page_number", 1)),
                },
            )
        )
    return docs


def ingest_clauses(
    policy_id: str,
    clauses: list[dict],
    *,
    db: Session | None = None,
) -> int:
    """Embed clauses and upsert into policy_clauses. Returns row count written."""
    docs = clauses_to_documents(policy_id, clauses)
    # Identity "chunking": fixtures are already one clause per document.
    texts = [d.text for d in docs]
    vectors = embed_texts(texts)

    owns_session = db is None
    if owns_session:
        db_session.ensure_engine()
        assert db_session.SessionLocal is not None
        db = db_session.SessionLocal()

    assert db is not None
    try:
        db.execute(delete(PolicyClause).where(PolicyClause.policy_id == policy_id))
        for doc, vector in zip(docs, vectors, strict=True):
            if not vector:
                raise RuntimeError(f"Empty embedding for clause {doc.metadata['clause_id']}")
            db.add(
                PolicyClause(
                    id=uuid.uuid4(),
                    policy_id=policy_id,
                    clause_id=doc.metadata["clause_id"],
                    clause_type=doc.metadata["clause_type"],
                    text=doc.text,
                    embedding=vector,
                    page_number=int(doc.metadata["page_number"]),
                )
            )
        db.commit()
        logger.info("Ingested %s clauses for policy_id=%s", len(docs), policy_id)
        return len(docs)
    except Exception:
        db.rollback()
        raise
    finally:
        if owns_session:
            db.close()


def ingest_file(path: Path, *, db: Session | None = None) -> int:
    policy_id, clauses = load_clause_file(path)
    return ingest_clauses(policy_id, clauses, db=db)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest policy clause JSON into pgvector")
    parser.add_argument("json_path", type=Path, help="Path to clause fixture JSON")
    args = parser.parse_args(argv)

    db_session.init_db()
    count = ingest_file(args.json_path)
    print(f"Ingested {count} clauses from {args.json_path}")


if __name__ == "__main__":
    main()
