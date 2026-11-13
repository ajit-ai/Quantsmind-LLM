"""Opt-in live provider qualification.

These tests are skipped unless the operator explicitly enables them. They use a
harmless minimal prompt and never transmit repository content.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.coding_providers import OllamaProvider, OpenAICompatibleProvider, ProviderConfig
from serving.providers import GenerationConfig, GenerationRequest

LIVE_PROMPT = "Reply with the single word OK."


@pytest.mark.skipif(
    os.environ.get("TAKSH_LIVE_OPENAI") != "1",
    reason="set TAKSH_LIVE_OPENAI=1 with explicit credentials to run",
)
def test_live_openai_compatible_minimal_prompt():
    config = ProviderConfig.from_env().get("openai-compatible")
    assert config is not None, "TAKSH_OPENAI_API_KEY is required"
    provider = OpenAICompatibleProvider(config=config)
    result = provider.generate(
        GenerationRequest(LIVE_PROMPT, GenerationConfig(max_new_tokens=8, timeout_seconds=60))
    )
    assert result.text.strip()
    assert result.provider == "openai-compatible"
    assert config.api_key not in result.text


@pytest.mark.skipif(
    os.environ.get("TAKSH_LIVE_OLLAMA") != "1",
    reason="set TAKSH_LIVE_OLLAMA=1 with a running local model to run",
)
def test_live_ollama_minimal_prompt():
    config = ProviderConfig.from_env().get("ollama")
    assert config is not None and config.model, "TAKSH_OLLAMA_MODEL is required"
    provider = OllamaProvider(config=config)
    probe = provider.probe()
    if not probe["model_present"]:
        pytest.skip(f"ollama model {config.model!r} is not present locally")
    result = provider.generate(
        GenerationRequest(LIVE_PROMPT, GenerationConfig(max_new_tokens=8, timeout_seconds=60))
    )
    assert result.text.strip()
    assert result.provider == "ollama"
