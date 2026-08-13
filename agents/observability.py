"""Optional Langfuse observability (Slice 8).

When LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are unset, all helpers are
no-ops so offline pytest and CI stay green. Tracing failures never fail a claim.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from api.config import get_settings

logger = logging.getLogger(__name__)

# Test injection point — set to a FakeLangfuse-like object in unit tests.
_client_override: Any | None = None


def set_client_override(client: Any | None) -> None:
    """Override the Langfuse client (tests only). Pass None to clear."""
    global _client_override
    _client_override = client


def langfuse_enabled() -> bool:
    if _client_override is not None:
        return True
    return get_settings().langfuse_enabled


def get_client() -> Any | None:
    """Return a Langfuse client, injected override, or None when disabled."""
    if _client_override is not None:
        return _client_override
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:  # noqa: BLE001 — never break the pipeline for obs
        logger.exception("Failed to construct Langfuse client")
        return None


def parse_openai_usage(data: dict[str, Any] | None) -> dict[str, int] | None:
    """Extract token counts from an OpenAI Chat Completions JSON body."""
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    out: dict[str, int] = {}
    for src, dest in (
        ("prompt_tokens", "input"),
        ("completion_tokens", "output"),
        ("total_tokens", "total"),
    ):
        val = usage.get(src)
        if isinstance(val, int):
            out[dest] = val
    return out or None


@contextmanager
def trace_claim(claim_id: str) -> Iterator[Any | None]:
    """Root span for process_claim, tagged with claim_id. No-op when disabled."""
    client = get_client()
    if client is None:
        yield None
        return
    try:
        start = getattr(client, "start_as_current_span", None) or getattr(
            client, "start_as_current_observation", None
        )
        if start is None:
            yield None
            return
        kwargs: dict[str, Any] = {
            "name": "process_claim",
            "metadata": {"claim_id": claim_id},
            "input": {"claim_id": claim_id},
        }
        # v3 observation API uses as_type=; v2 span API ignores unknown kwargs carefully
        if start.__name__ == "start_as_current_observation":
            kwargs["as_type"] = "span"
        with start(**kwargs) as root:
            try:
                update_trace = getattr(client, "update_current_trace", None)
                if callable(update_trace):
                    update_trace(
                        session_id=claim_id,
                        metadata={"claim_id": claim_id},
                        tags=["claimsight", "process_claim"],
                    )
            except Exception:  # noqa: BLE001
                logger.debug("Langfuse update_current_trace failed", exc_info=True)
            yield root
        flush = getattr(client, "flush", None)
        if callable(flush):
            flush()
    except Exception:  # noqa: BLE001
        logger.exception("Langfuse trace_claim failed; continuing without tracing")
        yield None


@contextmanager
def span(
    name: str,
    *,
    input: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Child span under the current claim trace. No-op when disabled / no client."""
    client = get_client()
    if client is None:
        yield None
        return
    try:
        start = getattr(client, "start_as_current_span", None) or getattr(
            client, "start_as_current_observation", None
        )
        if start is None:
            yield None
            return
        kwargs: dict[str, Any] = {"name": name}
        if input is not None:
            kwargs["input"] = input
        if metadata:
            kwargs["metadata"] = metadata
        if start.__name__ == "start_as_current_observation":
            kwargs["as_type"] = "span"
        with start(**kwargs) as child:
            yield child
    except Exception:  # noqa: BLE001
        logger.exception("Langfuse span %s failed; continuing", name)
        yield None


def update_span(observation: Any | None, **kwargs: Any) -> None:
    """Best-effort update of a span/generation (output, metadata, …)."""
    if observation is None:
        return
    try:
        updater = getattr(observation, "update", None)
        if callable(updater):
            updater(**kwargs)
    except Exception:  # noqa: BLE001
        logger.debug("Langfuse span update failed", exc_info=True)


@contextmanager
def generation(
    name: str,
    *,
    model: str | None = None,
    input: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """LLM generation observation nested under the current span."""
    client = get_client()
    if client is None:
        yield None
        return
    try:
        start_obs = getattr(client, "start_as_current_observation", None)
        start_gen = getattr(client, "start_as_current_generation", None)
        if start_obs is not None:
            kwargs: dict[str, Any] = {"name": name, "as_type": "generation"}
            if model is not None:
                kwargs["model"] = model
            if input is not None:
                kwargs["input"] = input
            if metadata:
                kwargs["metadata"] = metadata
            with start_obs(**kwargs) as gen:
                yield gen
            return
        if start_gen is not None:
            kwargs = {"name": name}
            if model is not None:
                kwargs["model"] = model
            if input is not None:
                kwargs["input"] = input
            if metadata:
                kwargs["metadata"] = metadata
            with start_gen(**kwargs) as gen:
                yield gen
            return
        yield None
    except Exception:  # noqa: BLE001
        logger.exception("Langfuse generation %s failed; continuing", name)
        yield None


def record_generation_usage(
    observation: Any | None,
    *,
    output: str | None = None,
    usage: dict[str, int] | None = None,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Attach output + token usage to a generation observation."""
    if observation is None:
        return
    payload: dict[str, Any] = {}
    if output is not None:
        payload["output"] = output
    if model is not None:
        payload["model"] = model
    if metadata:
        payload["metadata"] = metadata
    if usage:
        # Langfuse v2 uses `usage=` with prompt/completion/total tokens;
        # v3 prefers usage_details with input/output/total.
        payload["usage"] = {
            "input": usage.get("input"),
            "output": usage.get("output"),
            "total": usage.get("total"),
            "unit": "TOKENS",
        }
        payload["usage_details"] = {
            k: v for k, v in usage.items() if v is not None
        }
    try:
        updater = getattr(observation, "update", None)
        if callable(updater):
            # Drop keys the SDK version may not accept by retrying thinner payloads.
            try:
                updater(**payload)
            except TypeError:
                thin = {k: v for k, v in payload.items() if k in {"output", "model", "usage", "metadata"}}
                updater(**thin)
    except Exception:  # noqa: BLE001
        logger.debug("Langfuse record_generation_usage failed", exc_info=True)
