import pytest

gradio = pytest.importorskip("gradio")

from serving.app import build_ui


def test_build_ui_returns_blocks():
    ui = build_ui()
    assert ui is not None
    assert ui.__class__.__name__ in ("Blocks", "ChatInterface")
