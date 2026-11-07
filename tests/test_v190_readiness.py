"""Taksh v1.9.0 — GPU Readiness Tests.

Tests that verify the system is ready for GPU training.
These tests can run on CPU but will skip GPU-specific tests if no GPU is available.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Required files for v1.9.0 training
REQUIRED_FILES = [
    "configs/model/code_v170_100m.yaml",
    "configs/train/pretrain_v170_100m.yaml",
    "configs/train/sft_v170.yaml",
    "tokenizer/taksh_code_bpe_v1.json",
    "data/v170_tokens/train.tokens.bin",
    "data/v170_tokens/val.tokens.bin",
    "data/v170_tokens/train.tokens.meta.json",
    "data/v170_tokens/val.tokens.meta.json",
    "data/v170_tokens/manifest.json",
    "training/pretrain.py",
    "training/sft_train.py",
    "training/model.py",
    "training/checkpoint.py",
    "serving/local_inference.py",
    "serving/server_security.py",
]


class TestV190FileExistence:
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


class TestV190ConfigValidity:
    """Configuration files must be valid and consistent."""

    def test_model_config_has_required_keys(self):
        import yaml
        cfg_path = REPO / "configs/model/code_v170_100m.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
        required_keys = ["vocab_size", "d_model", "n_layers", "n_heads",
                         "n_kv_heads", "ffn_dim", "max_seq_len"]
        for key in required_keys:
            assert key in model_cfg, f"Missing model config key: {key}"

    def test_model_config_divisibility(self):
        import yaml
        cfg_path = REPO / "configs/model/code_v170_100m.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
        # d_model must be divisible by n_heads
        assert model_cfg["d_model"] % model_cfg["n_heads"] == 0
        # n_heads must be divisible by n_kv_heads
        assert model_cfg["n_heads"] % model_cfg["n_kv_heads"] == 0

    def test_training_config_has_required_keys(self):
        import yaml
        cfg_path = REPO / "configs/train/pretrain_v170_100m.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        required_keys = ["run", "data", "optimizer", "precision", "batch", "training"]
        for key in required_keys:
            assert key in cfg, f"Missing training config key: {key}"

    def test_training_max_steps_positive(self):
        import yaml
        cfg_path = REPO / "configs/train/pretrain_v170_100m.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        train_cfg = cfg.get("training", {})
        assert "max_steps" in train_cfg
        assert isinstance(train_cfg["max_steps"], int)
        assert train_cfg["max_steps"] > 0

    def test_seq_len_not_exceeding_model_max(self):
        import yaml
        model_cfg_path = REPO / "configs/model/code_v170_100m.yaml"
        train_cfg_path = REPO / "configs/train/pretrain_v170_100m.yaml"
        with model_cfg_path.open("r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f).get("model", {})
        with train_cfg_path.open("r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f)
        model_max_seq = model_cfg.get("max_seq_len", 0)
        train_seq = train_cfg.get("data", {}).get("seq_len", 0)
        assert train_seq <= model_max_seq, (
            f"data.seq_len ({train_seq}) exceeds model.max_seq_len ({model_max_seq})"
        )


class TestV190Tokenizer:
    """Tokenizer must be valid and match model config."""

    def test_tokenizer_loads(self):
        from tokenizers import Tokenizer
        tok_path = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
        assert tok_path.exists(), "Tokenizer file not found"
        tok = Tokenizer.from_file(str(tok_path))
        assert tok is not None

    def test_tokenizer_vocab_matches_model(self):
        import yaml
        from tokenizers import Tokenizer
        model_cfg_path = REPO / "configs/model/code_v170_100m.yaml"
        with model_cfg_path.open("r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f).get("model", {})
        model_vocab = model_cfg.get("vocab_size")
        tok_path = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
        tok = Tokenizer.from_file(str(tok_path))
        tok_vocab = tok.get_vocab_size()
        assert model_vocab == tok_vocab, (
            f"Model vocab_size ({model_vocab}) != tokenizer vocab_size ({tok_vocab})"
        )

    def test_tokenizer_has_special_tokens(self):
        from tokenizers import Tokenizer
        tok_path = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
        tok = Tokenizer.from_file(str(tok_path))
        for token in ["<pad>", "<bos>", "<eos>", "unk"]:
            assert tok.token_to_id(token) is not None, f"Missing special token: {token}"


class TestV190DataIntegrity:
    """Data files must be valid and consistent."""

    def test_token_files_exist(self):
        train_bin = REPO / "data" / "v170_tokens" / "train.tokens.bin"
        val_bin = REPO / "data" / "v170_tokens" / "val.tokens.bin"
        assert train_bin.exists(), "Train tokens file missing"
        assert val_bin.exists(), "Val tokens file missing"

    def test_token_files_non_empty(self):
        import numpy as np
        train_bin = REPO / "data" / "v170_tokens" / "train.tokens.bin"
        val_bin = REPO / "data" / "v170_tokens" / "val.tokens.bin"
        train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
        val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
        assert len(train_data) > 0, "Train tokens file is empty"
        assert len(val_data) > 0, "Val tokens file is empty"

    def test_token_metadata_consistency(self):
        import json
        import numpy as np
        for split in ["train", "val"]:
            bin_path = REPO / "data" / "v170_tokens" / f"{split}.tokens.bin"
            meta_path = REPO / "data" / "v170_tokens" / f"{split}.tokens.meta.json"
            assert meta_path.exists(), f"Meta file missing for {split}"
            data = np.memmap(bin_path, dtype=np.uint16, mode="r")
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            assert len(data) == meta.get("num_tokens"), (
                f"Token count mismatch for {split}: {len(data)} vs {meta.get('num_tokens')}"
            )


class TestV190Environment:
    """Environment must have required dependencies."""

    def test_required_packages_installed(self):
        # Map package name -> import name
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
        # This test always passes but reports GPU status
        print(f"\nGPU available: {cuda_available}")
        if cuda_available:
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("No GPU available - training will be CPU-only (slow)")


class TestV190TrainingImport:
    """Training modules must be importable."""

    def test_pretrain_importable(self):
        from training import pretrain
        from serving import local_inference
        assert pretrain is not None
        assert hasattr(pretrain, "main")
        assert local_inference is not None
        assert hasattr(local_inference, "load_model")
        assert hasattr(local_inference, "generate_ids")

    def test_sft_train_importable(self):
        from training import sft_train
        assert sft_train is not None
        assert hasattr(sft_train, "main")
        assert hasattr(sft_train, "run")

    def test_model_importable(self):
        from training import model
        assert model is not None
        assert hasattr(model, "QMModel")

    def test_checkpoint_importable(self):
        from training import checkpoint
        assert checkpoint is not None
        assert hasattr(checkpoint, "load_model_weights")


class TestV190ServingImport:
    """Serving modules must be importable."""

    def test_local_inference_importable(self):
        from serving import local_inference
        assert local_inference is not None
        assert hasattr(local_inference, "load_model")
        assert hasattr(local_inference, "generate_ids")

    def test_server_security_importable(self):
        from serving import server_security
        assert server_security is not None


class TestV190TorchFunctionality:
    """Basic torch functionality must work."""

    def test_tensor_creation(self):
        import torch
        x = torch.tensor([1, 2, 3])
        assert x.shape == (3,)

    def test_model_creation(self):
        import torch
        from training.model import QMModel
        cfg = {
            "vocab_size": 5822,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_dim": 256,
            "max_seq_len": 512,
            "rope_theta": 10000.0,
            "norm_eps": 1e-5,
            "tie_embeddings": True,
        }
        model = QMModel(cfg)
        assert model is not None
        assert model.num_params() > 0

    def test_forward_pass(self):
        import torch
        from training.model import QMModel
        cfg = {
            "vocab_size": 5822,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_dim": 256,
            "max_seq_len": 512,
            "rope_theta": 10000.0,
            "norm_eps": 1e-5,
            "tie_embeddings": True,
        }
        model = QMModel(cfg).eval()
        x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)
        assert logits.shape == (1, 5, 5822)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
