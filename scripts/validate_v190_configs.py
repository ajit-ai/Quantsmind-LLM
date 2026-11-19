"""Taksh v1.9.0 — Configuration Validator.

Validates that all v1.9.0 configuration files are consistent, complete,
and ready for GPU training.

Usage:
    python scripts/validate_v190_configs.py
    python scripts/validate_v190_configs.py --fix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent

CONFIG_FILES = [
    "configs/model/code_v170_100m.yaml",
    "configs/train/pretrain_v170_100m.yaml",
    "configs/train/sft_v170.yaml",
]

CROSS_CONFIG_CHECKS = [
    # (config_file, key, expected_value_in_other_config, other_config_file, other_key)
    # vocab_size must match between model config and training config
    ("configs/model/code_v170_100m.yaml", "vocab_size", "vocab_size", "configs/train/pretrain_v170_100m.yaml", None),
]


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_model_config(cfg: Dict[str, Any]) -> List[str]:
    """Validate model configuration."""
    errors = []
    required_keys = ["name", "vocab_size", "d_model", "n_layers", "n_heads",
                     "n_kv_heads", "ffn_dim", "max_seq_len", "rope_theta",
                     "norm_eps", "tie_embeddings"]

    for key in required_keys:
        if key not in cfg.get("model", {}):
            errors.append(f"Missing model config key: model.{key}")

    model_cfg = cfg.get("model", {})

    # Check divisibility constraints
    if "d_model" in model_cfg and "n_heads" in model_cfg:
        if model_cfg["d_model"] % model_cfg["n_heads"] != 0:
            errors.append(f"d_model ({model_cfg['d_model']}) must be divisible by n_heads ({model_cfg['n_heads']})")

    if "n_heads" in model_cfg and "n_kv_heads" in model_cfg:
        if model_cfg["n_heads"] % model_cfg["n_kv_heads"] != 0:
            errors.append(f"n_heads ({model_cfg['n_heads']}) must be divisible by n_kv_heads ({model_cfg['n_kv_heads']})")

    # Check reasonable ranges
    if "d_model" in model_cfg and model_cfg["d_model"] < 64:
        errors.append(f"d_model ({model_cfg['d_model']}) is too small (minimum 64)")

    if "n_layers" in model_cfg and model_cfg["n_layers"] < 1:
        errors.append(f"n_layers ({model_cfg['n_layers']}) must be >= 1")

    if "max_seq_len" in model_cfg and model_cfg["max_seq_len"] < 128:
        errors.append(f"max_seq_len ({model_cfg['max_seq_len']}) is too small (minimum 128)")

    return errors


def validate_training_config(cfg: Dict[str, Any]) -> List[str]:
    """Validate training configuration."""
    errors = []
    required_keys = ["run", "data", "optimizer", "precision", "batch", "training"]

    for key in required_keys:
        if key not in cfg:
            errors.append(f"Missing top-level key: {key}")

    train_cfg = cfg.get("training", {})
    if "max_steps" not in train_cfg:
        errors.append("Missing training.max_steps")
    elif not isinstance(train_cfg["max_steps"], int) or train_cfg["max_steps"] <= 0:
        errors.append(f"training.max_steps must be positive integer, got {train_cfg['max_steps']}")

    batch_cfg = cfg.get("batch", {})
    if "micro_batch_size" not in batch_cfg:
        errors.append("Missing batch.micro_batch_size")
    elif not isinstance(batch_cfg["micro_batch_size"], int) or batch_cfg["micro_batch_size"] <= 0:
        errors.append(f"batch.micro_batch_size must be positive integer, got {batch_cfg['micro_batch_size']}")

    data_cfg = cfg.get("data", {})
    if "seq_len" not in data_cfg:
        errors.append("Missing data.seq_len")
    elif "max_seq_len" in cfg.get("model", {}):
        if data_cfg["seq_len"] > cfg["model"]["max_seq_len"]:
            errors.append(f"data.seq_len ({data_cfg['seq_len']}) exceeds model.max_seq_len ({cfg['model']['max_seq_len']})")

    return errors


def validate_cross_config(configs: Dict[str, Dict[str, Any]]) -> List[str]:
    """Validate consistency across config files."""
    errors = []

    # vocab_size consistency
    model_cfg = configs.get("configs/model/code_v170_100m.yaml", {}).get("model", {})
    train_cfg = configs.get("configs/train/pretrain_v170_100m.yaml", {})

    model_vocab = model_cfg.get("vocab_size")
    # Check tokenizer vocab matches model vocab
    tokenizer_path = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
    if tokenizer_path.exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer_path))
        tok_vocab = tok.get_vocab_size()
        if model_vocab != tok_vocab:
            errors.append(f"Model vocab_size ({model_vocab}) != tokenizer vocab_size ({tok_vocab})")

    # seq_len consistency
    model_max_seq = model_cfg.get("max_seq_len")
    train_seq = train_cfg.get("data", {}).get("seq_len")
    if model_max_seq and train_seq and train_seq > model_max_seq:
        errors.append(f"data.seq_len ({train_seq}) > model.max_seq_len ({model_max_seq})")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="v1.9.0 Configuration Validator")
    parser.add_argument("--fix", action="store_true", help="Attempt to auto-fix issues")
    args = parser.parse_args()

    print("=" * 60)
    print("TAKSH v1.9.0 Configuration Validator")
    print("=" * 60)
    print()

    all_errors = []
    configs = {}

    # Load and validate each config
    for config_path in CONFIG_FILES:
        path = REPO / config_path
        if not path.exists():
            print(f"MISSING: {config_path}")
            all_errors.append(f"Missing config file: {config_path}")
            continue

        try:
            import yaml
            cfg = load_yaml(path)
            configs[config_path] = cfg
        except Exception as exc:
            print(f"ERROR loading {config_path}: {exc}")
            all_errors.append(f"Error loading {config_path}: {exc}")
            continue

        # Validate model config
        if config_path.endswith(".yaml") and "model" in config_path:
            model_errors = validate_model_config(cfg)
            if model_errors:
                print(f"MODEL CONFIG ERRORS in {config_path}:")
                for e in model_errors:
                    print(f"  - {e}")
                all_errors.extend([f"{config_path}: {e}" for e in model_errors])
            else:
                print(f"OK: {config_path} (model config valid)")

        # Validate training config
        if "training" in cfg:
            train_errors = validate_training_config(cfg)
            if train_errors:
                print(f"TRAINING CONFIG ERRORS in {config_path}:")
                for e in train_errors:
                    print(f"  - {e}")
                all_errors.extend([f"{config_path}: {e}" for e in train_errors])
            else:
                print(f"OK: {config_path} (training config valid)")

    # Cross-config validation
    print()
    print("Cross-config validation:")
    cross_errors = validate_cross_config(configs)
    if cross_errors:
        for e in cross_errors:
            print(f"  CROSS-CONFIG ERROR: {e}")
        all_errors.extend(cross_errors)
    else:
        print("  All cross-config checks passed")

    print()
    print("=" * 60)
    if all_errors:
        print(f"FOUND {len(all_errors)} ERROR(S)")
        print("Fix these before training:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL CONFIGURATIONS VALID")
        print("Ready for training")


if __name__ == "__main__":
    main()
