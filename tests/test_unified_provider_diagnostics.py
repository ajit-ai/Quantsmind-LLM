"""Focused diagnostics checks for the unified provider registry."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.provider_diagnostics import diagnostic, diagnostics


def test_diagnostics_lists_unified_providers():
    names = set(diagnostics())
    assert "openai-compatible" in names
    assert "ollama" in names
    assert "deterministic-mock" in names
    assert "native-taksh" in names


def test_unknown_provider_is_explicit():
    report = diagnostic("not-a-provider")
    assert report["status"] == "unknown_provider"
    assert report["available"] is False


def test_deterministic_provider_is_labelled_test_only():
    report = diagnostic("deterministic-mock")
    assert report["status"] == "test_only"
    assert report["test_only"] is True
    assert report["available"] is True


def test_native_provider_reports_missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("TAKSH_NATIVE_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("TAKSH_NATIVE_TOKENIZER", str(tmp_path / "missing-tokenizer.json"))
    report = diagnostic("native-taksh")
    assert report["status"] == "checkpoint_missing"
    assert report["available"] is False


def test_openai_unconfigured_is_unauthenticated(monkeypatch):
    monkeypatch.delenv("TAKSH_OPENAI_API_KEY", raising=False)
    report = diagnostic("openai-compatible")
    assert report["status"] == "unauthenticated"
    assert report["available"] is False


def test_credentials_are_never_returned(monkeypatch):
    secret = "top-secret-provider-key"
    monkeypatch.setenv("TAKSH_OPENAI_API_KEY", secret)
    monkeypatch.setenv("TAKSH_OPENAI_MODEL", "test-model")
    report = diagnostic("openai-compatible")
    serialized = repr(report)
    assert secret not in serialized
    assert "api_key" not in report
    assert report["status"] == "configured"


def test_diagnostics_do_not_contact_networks(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostics must not perform network I/O")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    report = diagnostics()
    assert report
    with pytest.raises(AssertionError, match="must not perform network I/O"):
        urllib.request.urlopen("https://example.invalid")
