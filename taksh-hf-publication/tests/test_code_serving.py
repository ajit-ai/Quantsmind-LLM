import pytest

gradio = pytest.importorskip("gradio")

from serving.code_chat import format_prompt, LANGUAGE_INFO


def test_format_prompt_basic():
    prompt = format_prompt("Write a function", "python")
    assert "Write a function" in prompt
    assert "Python" in prompt


def test_format_prompt_with_context():
    prompt = format_prompt("Review this", "python", context="def foo(): pass")
    assert "Review this" in prompt
    assert "def foo(): pass" in prompt
    assert "Context" in prompt


def test_format_prompt_javascript():
    prompt = format_prompt("Create a component", "javascript")
    assert "Create a component" in prompt
    assert "JavaScript" in prompt


def test_language_info_structure():
    assert "python" in LANGUAGE_INFO
    assert "javascript" in LANGUAGE_INFO
    assert "typescript" in LANGUAGE_INFO
    for lang, info in LANGUAGE_INFO.items():
        assert "extensions" in info
        assert "prompt_prefix" in info
