"""Celery application configuration."""

from __future__ import annotations

from celery import Celery

from api.config import get_settings

settings = get_settings()

celery_app = Celery(
    "claimsight",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=True,
)
