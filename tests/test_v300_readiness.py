"""Taksh v3.0.0 — Advanced/Enterprise Readiness Tests.

Tests that verify the v3.0.0 configuration and infrastructure is ready
for GPU training. These tests can run on CPU but will skip GPU-specific
tests if no GPU is available.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Required files for v3.0.0 training
REQUIRED_FILES = [
    "configs/model/code_v300_70b.yaml",
    "configs/model/code_v300_30b.yaml",
    "configs/train/pretrain_v300_70b.yaml",
    "configs/train/sft_v300_70b.yaml",
    "configs/train/dpo_v300_70b.yaml",
    "configs/tokenizer/code_bpe_v3.yaml",
    "configs/eval/v300_multimodal.yaml",
    "training/pretrain.py",
    "training/sft_train.py",
    "training/model.py",
    "training/checkpoint.py",
    "scripts/eval_v300_multimodal.py",
]


class TestV300FileExistence:
    """All required files must exist for training."""

    def test_all_required_files_exist(self):
        missing = []
        for f in REQUIRED_FILES:
            if not (REPO / f).exists():
                missing.append(f)
        assert not missing, f"Missing required files: {missing}"

    @pytest.mark.parametrize("filename", REQUIRED_FILES)
    def test_individual_file_exists(self, filename):
        assert (REPO / filename).exists(), f"Missing required file: {filename}"


class TestV300ConfigValidity:
    """Configuration files must be valid and consistent."""

    def test_model_config_has_required_keys(self):
        import yaml

        cfg_path = REPO / "configs/model/code_v300_70b.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
        required_keys = [
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "ffn_dim",
            "max_seq_len",
        ]
        for key in required_keys:
            assert key in model_cfg, f"Missing model config key: {key}"

    def test_model_config_divisibility(self):
        import yaml

        cfg_path = REPO / "configs/model/code_v300_70b.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
        assert model_cfg["d_model"] % model_cfg["n_heads"] == 0
        assert model_cfg["n_heads"] % model_cfg["n_kv_heads"] == 0

    def test_training_config_has_required_keys(self):
        import yaml

        cfg_path = REPO / "configs/train/pretrain_v300_70b.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        required_keys = ["run", "data", "optimizer", "precision", "batch", "training"]
        for key in required_keys:
            assert key in cfg, f"Missing training config key: {key}"

    def test_training_max_steps_positive(self):
        import yaml

        cfg_path = REPO / "configs/train/pretrain_v300_70b.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        train_cfg = cfg.get("training", {})
        assert "max_steps" in train_cfg
        assert isinstance(train_cfg["max_steps"], int)
        assert train_cfg["max_steps"] > 0

    def test_seq_len_not_exceeding_model_max(self):
        import yaml

        model_cfg_path = REPO / "configs/model/code_v300_70b.yaml"
        train_cfg_path = REPO / "configs/train/pretrain_v300_70b.yaml"
        with model_cfg_path.open("r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f).get("model", {})
        with train_cfg_path.open("r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f)
        model_max_seq = model_cfg.get("max_seq_len", 0)
        train_seq = train_cfg.get("data", {}).get("seq_len", 0)
        assert train_seq <= model_max_seq, (
            f"data.seq_len ({train_seq}) exceeds model.max_seq_len ({model_max_seq})"
        )


class TestV300Tokenizer:
    """Tokenizer must be valid and match model config."""

    def test_tokenizer_config_has_required_keys(self):
        import yaml

        cfg_path = REPO / "configs/tokenizer/code_bpe_v3.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "tokenizer" in cfg
        assert cfg["tokenizer"]["vocab_size"] == 65536

    def test_tokenizer_file_exists(self):
        from pathlib import Path

        tok_path = REPO / "tokenizer" / "taksh_code_bpe_v3.json"
        assert tok_path.exists(), "Tokenizer file not found"


class TestV300Environment:
    """Environment must have required dependencies."""

    def test_required_packages_installed(self):
        required_packages = [
            ("torch", "torch"),
            ("tokenizers", "tokenizers"),
            ("numpy", "numpy"),
            ("pyyaml", "yaml"),
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"),
            ("pydantic", "pydantic"),
            ("gradio", "gradio"),
            ("pytest", "pytest"),
        ]
        missing = []
        for pkg_name, import_name in required_packages:
            try:
                importlib.import_module(import_name)
            except ImportError:
                missing.append(pkg_name)
        assert not missing, f"Missing packages: {missing}"

    def test_torch_importable(self):
        import torch

        assert torch is not None

    def test_gpu_availability_checked(self):
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"\nGPU available: {cuda_available}")
        if cuda_available:
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("No GPU available - training will be CPU-only (slow)")


class TestV300TrainingImport:
    """Training modules must be importable."""

    def test_pretrain_importable(self):
        from training import pretrain

        assert pretrain is not None
        assert hasattr(pretrain, "main")

    def test_sft_train_importable(self):
        from training import sft_train

        assert sft_train is not None
        assert hasattr(sft_train, "run")

    def test_model_importable(self):
        from training import model

        assert model is not None
        assert hasattr(model, "QMModel")

    def test_checkpoint_importable(self):
        from training import checkpoint

        assert checkpoint is not None
        assert hasattr(checkpoint, "load_model_weights")


class TestV300ServingImport:
    """Serving modules must be importable."""

    def test_local_inference_importable(self):
        from serving import local_inference

        assert local_inference is not None
        assert hasattr(local_inference, "load_model")
        assert hasattr(local_inference, "generate_ids")

    def test_server_security_importable(self):
        from serving import server_security

        assert server_security is not None


class TestV300TorchFunctionality:
    """Basic torch functionality must work."""

    def test_tensor_creation(self):
        import torch

        x = torch.tensor([1, 2, 3])
        assert x.shape == (3,)

    def test_model_creation(self):
        import torch
        from training.model import QMModel

        cfg = {
            "vocab_size": 32768,
            "d_model": 512,
            "n_layers": 4,
            "n_heads": 8,
            "n_kv_heads": 2,
            "ffn_dim": 2048,
            "max_seq_len": 1024,
            "rope_theta": 1000000.0,
            "norm_eps": 1e-5,
            "tie_embeddings": False,
        }
        model = QMModel(cfg)
        assert model is not None
        assert model.num_params() > 0

    def test_forward_pass(self):
        import torch
        from training.model import QMModel

        cfg = {
            "vocab_size": 32768,
            "d_model": 512,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_dim": 2048,
            "max_seq_len": 512,
            "rope_theta": 1000000.0,
            "norm_eps": 1e-5,
            "tie_embeddings": False,
        }
        model = QMModel(cfg).eval()
        x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)
        assert logits.shape == (1, 5, 32768)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
