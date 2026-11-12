"""Synthetic checkpoint fixtures for tests that require geometry inspection.

The real ``runs/m7_pretrain/checkpoint_last.pt`` artifact is not tracked in
git (``runs/`` is gitignored), so a CI checkout has no production checkpoint.
This factory builds a *clearly identified synthetic* checkpoint with the exact
geometry of the qualified ``code_m3_9m`` configuration (vocab 5822, d_model
256, 8 layers, 8 heads, 4 KV heads, FFN 1024, max_seq_len 2048, tied
embeddings, RoPE θ=10000, RMSNorm, norm_eps 1e-5) and ``step=300`` metadata,
which is all the geometry-inspection and seeded-sampling tests consume.

It is NOT the production artifact, NOT a trained model, NOT a quality claim,
and must never be substituted for one. Deterministic construction (fixed seed)
keeps every test byte-stable across runs.
"""

from __future__ import annotations

from pathlib import Path

import torch

SMOKE_DICT = {
    "vocab_size": 5822,
    "d_model": 256,
    "n_layers": 8,
    "n_heads": 8,
    "n_kv_heads": 4,
    "ffn_dim": 1024,
    "max_seq_len": 2048,
    "rope_theta": 10000.0,
    "norm_eps": 1e-05,
    "tie_embeddings": True,
}


def build_s2_checkpoint(tmp_path: Path, *, step: int = 300) -> Path:
    """Build the synthetic S2-equivalent checkpoint under ``tmp_path``."""
    import sys

    REPO = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO))

    from training.model import QMModel

    torch.manual_seed(20261111)
    model = QMModel(SMOKE_DICT)
    path = tmp_path / "s2_synthetic_checkpoint.pt"
    torch.save({
        "model": model.state_dict(),
        "step": step,
        "config": {"run": {"model_config": str(REPO / "configs" / "model" / "code_m3_9m.yaml")}},
        "schema": "quantsmind.fixture/synthetic",
        "provenance": "synthetic test fixture — never a production artifact",
    }, path)
    return path
