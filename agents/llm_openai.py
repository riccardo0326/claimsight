"""Thin OpenAI Chat Completions client via httpx (Slice 5 Adjudicator).

No official OpenAI SDK — keeps the dependency surface aligned with Slice 4
external HTTP (httpx + Tenacity). See DECISIONS.md.

Slice 8: when a Langfuse trace context is active, records a generation with
token usage. complete_json still returns only the assistant content string.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agents.observability import (
    generation,
    parse_openai_usage,
    record_generation_usage,
)
from api.config import get_settings

logger = logging.getLogger(__name__)

PROMPT_VERSION = "prompts/adjudicator_v1.md"


class LLMError(Exception):
    """Raised when the frontier LLM call fails after retries."""


class LLMRetryableError(LLMError):
    """Transient LLM failure worth retrying (5xx / transport)."""


def complete_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Call Chat Completions and return the assistant message content (JSON text).

    Uses response_format json_object when supported by the endpoint.
    """
    settings = get_settings()
    key = api_key if api_key is not None else settings.openai_api_key
    if not key:
        raise LLMError("OPENAI_API_KEY is not configured")

    url_base = (base_url if base_url is not None else settings.adjudicator_base_url).rstrip(
        "/"
    )
    url = f"{url_base}/chat/completions"
    use_model = model if model is not None else settings.adjudicator_model
    use_timeout = (
        timeout if timeout is not None else settings.adjudicator_timeout_seconds
    )
    max_attempts = settings.external_api_max_attempts

    payload: dict[str, Any] = {
        "model": use_model,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": settings.http_user_agent,
    }

    owns_client = client is None
    http = client or httpx.Client()
    try:

        @retry(
            reraise=True,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.TransportError, LLMRetryableError)
            ),
        )
        def _once() -> tuple[str, dict[str, Any]]:
            try:
                resp = http.post(url, headers=headers, json=payload, timeout=use_timeout)
            except (httpx.TimeoutException, httpx.TransportError):
                raise
            if resp.status_code >= 500:
                raise LLMRetryableError(f"LLM HTTP {resp.status_code}")
            if resp.status_code >= 400:
                raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
            try:
                data = resp.json()
            except ValueError as exc:
                raise LLMError("LLM response is not JSON") from exc
            choices = data.get("choices") or []
            if not choices:
                raise LLMError("LLM response missing choices")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise LLMError("LLM response missing message content")
            return content, data if isinstance(data, dict) else {}

        started = time.perf_counter()
        with generation(
            "adjudicator_llm",
            model=use_model,
            input=messages,
            metadata={"prompt_version": PROMPT_VERSION},
        ) as gen:
            try:
                content, data = _once()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                logger.exception("Adjudicator LLM transport failure")
                raise LLMError(str(exc)) from exc
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            usage = parse_openai_usage(data)
            record_generation_usage(
                gen,
                output=content,
                usage=usage,
                model=use_model,
                metadata={
                    "prompt_version": PROMPT_VERSION,
                    "latency_ms": elapsed_ms,
                },
            )
            return content
    finally:
        if owns_client:
            http.close()
