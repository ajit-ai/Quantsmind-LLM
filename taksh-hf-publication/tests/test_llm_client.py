import pytest

from serving.llm_client import (
    LLMConfig,
    append_disclaimer,
    build_messages,
    demo_stream,
)


def test_configured_false_when_env_missing(monkeypatch):
    for var in ("QM_API_BASE", "QM_API_KEY", "QM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert not LLMConfig.from_env().configured()


def test_configured_true_when_env_set(monkeypatch):
    monkeypatch.setenv("QM_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setenv("QM_API_KEY", "test")
    monkeypatch.setenv("QM_MODEL", "qm-7b")
    assert LLMConfig.from_env().configured()


def test_append_disclaimer_once():
    out = append_disclaimer("Some answer")
    assert "not investment advice" in out.lower()
    assert append_disclaimer(out) == out


def test_build_messages_includes_system_and_history():
    history = [["What is P/E?", "Price to earnings."]]
    msgs = build_messages(history, "And EPS?")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "And EPS?"


def test_demo_stream_answers_pe():
    text = "".join(demo_stream("Explain P/E ratio"))
    assert "P/E" in text


def test_demo_stream_fallback():
    text = "".join(demo_stream("random gibberish xyz"))
    assert "demo mode" in text.lower()
