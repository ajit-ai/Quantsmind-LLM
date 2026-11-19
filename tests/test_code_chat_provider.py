import gradio as gr  # noqa: F401
import pytest

from serving.code_chat import (
    build_provider_card,
    format_prompt,
    generate_answer,
    render_note,
)
from serving.providers import GenerationRequest


class FakeIdentity:
    def __init__(self, **kwargs):
        self._kw = kwargs

    def as_dict(self):
        base = dict(
            provider="fake",
            model_id="fake-model",
            revision="rev1",
            source="memory",
            license_or_rights="Apache-2.0",
            artifact_sha256="00" * 32,
            tokenizer="tokenizer/taksh_code_bpe_v1.json",
        )
        base.update(self._kw)
        return base

    def validated(self):
        return self


class FakeCapabilities:
    def __init__(self, **kwargs):
        self._kw = kwargs

    def as_dict(self):
        base = dict(
            provider="fake",
            available=True,
            offline=True,
            configured=True,
            status="compatible",
            modes=("local",),
            max_context_tokens=2048,
            notes="fake provider",
        )
        base.update(self._kw)
        return base


class FakeProvider:
    name = "fake"

    def __init__(self, model_id="fake-model", status="compatible", raises=False):
        self._raises = raises
        self.identity_obj = FakeIdentity(model_id=model_id)
        self.caps_obj = FakeCapabilities(status=status)

    def identity(self):
        return self.identity_obj

    def capabilities(self):
        return self.caps_obj

    def generate(self, request: GenerationRequest):
        if self._raises:
            from serving.providers import ProviderUnavailableError

            raise ProviderUnavailableError("fake unavailable")
        return type("GenResult", (), {})()  # never used if we patch provider path


def test_format_prompt_basic_still_works():
    prompt = format_prompt("add two numbers", "python")
    assert "<bos>" in prompt
    assert "add two numbers" in prompt


def test_build_provider_card_is_honest_and_contains_source():
    provider = FakeProvider(model_id="taksh-native")
    card = build_provider_card("native-taksh", provider)
    assert "fake" in card
    assert "taksh-native" in card
    assert "Apache-2.0" in card


def test_build_provider_card_never_crashes_on_bad_provider():
    card = build_provider_card("native-taksh", object())
    assert "failed to build" in card.lower()


def test_generate_answer_reports_provider_honestly(monkeypatch, tmp_path):
    provider = FakeProvider()
    result_obj = type("R", (), {
        "provider": "fake",
        "model_id": "fake-model",
        "finish_reason": "stop",
        "text": "def f():\n    return 1\n",
    })()
    monkeypatch.setattr("serving.code_chat.generate_with_timeout", lambda provider_, request: result_obj)
    text = generate_answer(provider, "write it", "python", "", 0.0, 64, 1)
    assert "fake" in text
    assert "def f()" in text


def test_generate_answer_fail_closed_on_provider_error(monkeypatch):
    provider = FakeProvider(raises=True)
    text = generate_answer(provider, "write it", "python", "", 0.0, 64, 1)
    assert "failed" in text.lower()
    assert "No fallback" in text
