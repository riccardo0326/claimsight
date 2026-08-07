"""Shared embedding helpers for ingest and RAG (same model both sides)."""

from __future__ import annotations

import logging
from typing import Any

from api.config import get_settings

logger = logging.getLogger(__name__)

_embed_model: Any | None = None


def get_embed_model():
    """Lazy singleton LlamaIndex HuggingFaceEmbedding (MiniLM by default)."""
    global _embed_model
    if _embed_model is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        settings = get_settings()
        logger.info("Loading embedding model %s", settings.embedding_model)
        _embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
    return _embed_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embed_model()
    return [list(model.get_text_embedding(t)) for t in texts]


def embed_query(text: str) -> list[float]:
    model = get_embed_model()
    return list(model.get_query_embedding(text))
