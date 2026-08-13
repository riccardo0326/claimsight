"""Unit tests for Slice 8 Langfuse observability helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from agents.observability import (
    generation,
    langfuse_enabled,
    parse_openai_usage,
    record_generation_usage,
    set_client_override,
    span,
    trace_claim,
    update_span,
)


class _FakeObservation:
    def __init__(self, name: str, as_type: str = "span", **kwargs: Any):
        self.name = name
        self.as_type = as_type
        self.kwargs = kwargs
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeLangfuse:
    """Minimal stand-in that records span/generation names for tests."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.flushed = False
        self.trace_updates: list[dict[str, Any]] = []

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any):
        self.events.append(f"span:{name}")
        obs = _FakeObservation(name, as_type="span", **kwargs)
        yield obs

    @contextmanager
    def start_as_current_generation(self, name: str, **kwargs: Any):
        self.events.append(f"generation:{name}")
        obs = _FakeObservation(name, as_type="generation", **kwargs)
        yield obs

    def update_current_trace(self, **kwargs: Any) -> None:
        self.trace_updates.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


def test_parse_openai_usage_happy_path():
    usage = parse_openai_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
            }
        }
    )
    assert usage == {"input": 100, "output": 40, "total": 140}


def test_parse_openai_usage_missing():
    assert parse_openai_usage(None) is None
    assert parse_openai_usage({}) is None
    assert parse_openai_usage({"usage": {}}) is None


def test_noop_when_langfuse_unset(monkeypatch):
    set_client_override(None)
    monkeypatch.setattr(
        "agents.observability.get_settings",
        lambda: type(
            "S",
            (),
            {
                "langfuse_enabled": False,
                "langfuse_public_key": None,
                "langfuse_secret_key": None,
                "langfuse_host": "https://cloud.langfuse.com",
            },
        )(),
    )
    assert langfuse_enabled() is False
    with trace_claim("abc") as root:
        assert root is None
        with span("document") as child:
            assert child is None
            update_span(child, output={"ok": True})
        with generation("adjudicator_llm", model="gpt-4o") as gen:
            assert gen is None
            record_generation_usage(gen, output="{}", usage={"input": 1})


def test_fake_client_records_span_names():
    fake = FakeLangfuse()
    set_client_override(fake)
    try:
        with trace_claim("claim-1") as root:
            assert root is not None
            with span("document", input={"policy_pdf": "p.pdf"}) as doc:
                update_span(doc, output={"policy_id": "POL-1"})
            with span("vision", metadata={"skipped": True}):
                pass
            with generation("adjudicator_llm", model="gpt-4o") as gen:
                record_generation_usage(
                    gen,
                    output='{"decision":"approve"}',
                    usage={"input": 10, "output": 5, "total": 15},
                    model="gpt-4o",
                )
        assert fake.flushed is True
        assert "span:process_claim" in fake.events
        assert "span:document" in fake.events
        assert "span:vision" in fake.events
        assert "generation:adjudicator_llm" in fake.events
        assert fake.trace_updates
        assert fake.trace_updates[0].get("session_id") == "claim-1"
    finally:
        set_client_override(None)
