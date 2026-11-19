import pytest

from serving.llm_client import (
    DISCLAIMER,
    SYSTEM_PROMPT,
    ResidencyError,
    enforce_residency,
    is_private_endpoint,
)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:8000/v1", True),
        ("http://10.0.0.5:8000/v1", True),
        ("http://192.168.1.20:8000/v1", True),
        ("http://172.16.0.1:8000/v1", True),
        ("http://172.31.9.9:8000/v1", True),
        ("http://mymodel.local:8000/v1", True),
        ("http://172.32.0.1:8000/v1", False),
        ("https://api.openai.com/v1", False),
        ("https://openrouter.ai/api/v1", False),
        ("", False),
    ],
)
def test_is_private_endpoint(url, expected):
    assert is_private_endpoint(url) is expected


def test_enforce_residency_blocks_public(monkeypatch):
    monkeypatch.setenv("QM_STRICT_RESIDENCY", "1")
    with pytest.raises(ResidencyError):
        enforce_residency("https://api.openai.com/v1")


def test_enforce_residency_allows_private(monkeypatch):
    monkeypatch.setenv("QM_STRICT_RESIDENCY", "1")
    enforce_residency("http://localhost:11434/v1")
    enforce_residency("http://10.20.30.40:8000/v1")


def test_enforce_residency_override(monkeypatch):
    monkeypatch.setenv("QM_STRICT_RESIDENCY", "0")
    enforce_residency("https://api.openai.com/v1")


def test_system_prompt_india_specific():
    text = SYSTEM_PROMPT.lower()
    assert "sebi" in text
    assert "nse" in text and "bse" in text
    assert "crore" in text or "lakh" in text


def test_disclaimer_mentions_sebi_ria():
    assert "sebi-registered" in DISCLAIMER.lower()
