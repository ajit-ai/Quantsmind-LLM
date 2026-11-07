"""Taksh v2.2.0 — Specialized Domain Models Readiness Tests.

Tests that verify the v2.2.0 domain model configuration and infrastructure
is ready for GPU training. These tests can run on CPU.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Required files for v2.2.0 domain model training
REQUIRED_FILES = [
    "configs/model/code_v220_frontend_7b.yaml",
    "configs/model/code_v220_backend_7b.yaml",
    "configs/model/code_v220_datascience_7b.yaml",
    "configs/model/code_v220_embedded_7b.yaml",
    "configs/train/domain_v220_frontend_7b.yaml",
    "configs/train/domain_v220_backend_7b.yaml",
    "configs/train/domain_v220_datascience_7b.yaml",
    "configs/train/domain_v220_embedded_7b.yaml",
    "configs/tokenizer/code_bpe_v2.yaml",
    "configs/eval/v200_humaneval.yaml",
    "serving/domain_router.py",
    "scripts/eval_v220_domain.py",
    "training/pretrain.py",
    "training/sft_train.py",
    "training/model.py",
    "training/checkpoint.py",
]


class TestV220FileExistence:
    """All required files must exist for domain model training."""

    def test_all_required_files_exist(self):
        missing = []
        for f in REQUIRED_FILES:
            if not (REPO / f).exists():
                missing.append(f)
        assert not missing, f"Missing required files: {missing}"

    @pytest.mark.parametrize("filename", REQUIRED_FILES)
    def test_individual_file_exists(self, filename):
        assert (REPO / filename).exists(), f"Missing required file: {filename}"


class TestV220ConfigValidity:
    """Configuration files must be valid and consistent."""

    def test_domain_model_configs(self):
        import yaml

        for domain in ["frontend", "backend", "datascience", "embedded"]:
            cfg_path = REPO / f"configs/model/code_v220_{domain}_7b.yaml"
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
                assert key in model_cfg, (
                    f"Missing model config key: model.{key} in {domain}"
                )
            assert model_cfg["d_model"] % model_cfg["n_heads"] == 0
            assert model_cfg["n_heads"] % model_cfg["n_kv_heads"] == 0

    def test_domain_training_configs(self):
        import yaml

        for domain in ["frontend", "backend", "datascience", "embedded"]:
            cfg_path = REPO / f"configs/train/domain_v220_{domain}_7b.yaml"
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            required_keys = ["run", "data", "optimizer", "precision", "batch", "training"]
            for key in required_keys:
                assert key in cfg, (
                    f"Missing training config key: {key} in {domain}"
                )
            train_cfg = cfg.get("training", {})
            assert "max_steps" in train_cfg, f"Missing max_steps in {domain}"
            assert isinstance(train_cfg["max_steps"], int)
            assert train_cfg["max_steps"] > 0


class TestV220Tokenizer:
    """Tokenizer must be valid for domain models."""

    def test_tokenizer_config(self):
        import yaml

        cfg_path = REPO / "configs/tokenizer/code_bpe_v2.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "tokenizer" in cfg
        tokenizer_cfg = cfg["tokenizer"]
        assert tokenizer_cfg["vocab_size"] == 32768
        assert "special_tokens" in tokenizer_cfg
        assert len(tokenizer_cfg["special_tokens"]) > 0

    def test_tokenizer_file_exists(self):
        from pathlib import Path

        tok_path = REPO / "tokenizer" / "taksh_code_bpe_v2.json"
        assert tok_path.exists(), "Tokenizer file not found"


class TestV220Environment:
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


class TestV220TrainingImport:
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


class TestV220ServingImport:
    """Serving modules must be importable."""

    def test_domain_router_importable(self):
        from serving import domain_router

        assert domain_router is not None
        assert hasattr(domain_router, "DomainRouter")
        assert hasattr(domain_router, "route_request")

    def test_local_inference_importable(self):
        from serving import local_inference

        assert local_inference is not None
        assert hasattr(local_inference, "load_model")
        assert hasattr(local_inference, "generate_ids")

    def test_server_security_importable(self):
        from serving import server_security

        assert server_security is not None


class TestV220TorchFunctionality:
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
