"""Focused diagnostics checks for unified provider registry."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.provider_diagnostics import diagnostics


def test_diagnostics_lists_unified_providers():
    names = set(diagnostics())
    assert "openai-compatible" in names
    assert "ollama" in names
    assert "deterministic-mock" in names
    assert "native-taksh" in names
