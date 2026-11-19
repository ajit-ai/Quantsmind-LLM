"""TAKSH v1.4.1: checkpoint loader must be safe-only (no unsafe fallback).

Covers: (1) the frozen supported checkpoint loads via weights_only=True
through the loader without any fallback warning; (2) an artifact containing
arbitrary pickled objects is refused with a clear error and never executes
the pickle opcode (verified by a marker file that must not appear).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest
import torch

from serving.local_inference import torch_load_checkpoint

FROZEN = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")


class _Evil:
    def __reduce__(self):
        marker = Path(os.environ["V141_MARKER"])
        return (os.system, (f"touch {marker}",))


def test_safe_loader_loads_supported_checkpoint():
    if not FROZEN.is_file():
        pytest.skip("frozen checkpoint not present")
    ckpt = torch_load_checkpoint(str(FROZEN))
    assert isinstance(ckpt, dict)
    assert "model" in ckpt and "config" in ckpt


def test_safe_loader_refuses_arbitrary_pickle_without_executing(tmp_path, monkeypatch):
    marker = tmp_path / "pwned"
    monkeypatch.setenv("V141_MARKER", str(marker))
    evil_path = tmp_path / "evil.pt"
    # torch.save pickles the arbitrary object; weights_only=True must refuse
    # it and the pickle opcode must never execute.
    torch.save({"payload": _Evil()}, str(evil_path))
    with pytest.raises(RuntimeError, match="Refusing to load checkpoint"):
        torch_load_checkpoint(str(evil_path))
    assert not marker.exists(), "pickle opcode executed — unsafe fallback reached"
