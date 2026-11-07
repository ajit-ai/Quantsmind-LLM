"""Taksh v1.9.0 — Preflight validation & training preparation.

Validates that everything is ready for GPU training before launching.
Run this BEFORE starting pretraining on a GPU cluster.

Usage:
    python scripts/prepare_v190_training.py --check-all
    python scripts/prepare_v190_training.py --check-configs
    python scripts/prepare_v190_training.py --check-data
    python scripts/prepare_v190_training.py --check-environment
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent

# Expected files for v1.9.0 training
REQUIRED_FILES = [
    "configs/model/code_v170_100m.yaml",
    "configs/train/pretrain_v170_100m.yaml",
    "configs/train/sft_v170.yaml",
    "tokenizer/taksh_code_bpe_v1.json",
    "data/v170_tokens/train.tokens.bin",
    "data/v170_tokens/val.tokens.bin",
    "data/v170_tokens/train.tokens.bin.meta.json",
    "data/v170_tokens/val.tokens.bin.meta.json",
    "data/v170_tokens/manifest.json",
    "training/pretrain.py",
    "training/sft_train.py",
    "training/model.py",
    "training/checkpoint.py",
    "serving/local_inference.py",
    "serving/server_security.py",
]

EXPECTED_CONFIG_VALUES = {
    "configs/model/code_v170_100m.yaml": {
        "vocab_size": 5822,
        "max_seq_len": 4096,
        "d_model": 512,
        "n_layers": 16,
        "n_heads": 8,
        "n_kv_heads": 4,
        "ffn_dim": 2048,
    },
    "configs/train/pretrain_v170_100m.yaml": {
        "seq_len": 1024,
        "max_steps": 200,
    },
}


def check_file_exists(path: Path) -> Dict[str, Any]:
    """Check if a file exists and return details."""
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": size,
        "sha256": sha256,
    }


def check_config_values(config_path: Path, expected: Dict[str, Any]) -> Dict[str, Any]:
    """Check that config file contains expected values."""
    import yaml

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_cfg = cfg.get("model", cfg)
        results = {}
        all_match = True

        for key, expected_value in expected.items():
            actual = model_cfg.get(key)
            match = actual == expected_value
            if not match:
                all_match = False
            results[key] = {
                "expected": expected_value,
                "actual": actual,
                "match": match,
            }

        return {
            "config": str(config_path),
            "all_match": all_match,
            "checks": results,
        }
    except Exception as exc:
        return {
            "config": str(config_path),
            "all_match": False,
            "error": str(exc),
        }


def check_data_integrity() -> Dict[str, Any]:
    """Check that data files are valid and consistent."""
    results = []

    # Check train tokens
    train_bin = REPO / "data" / "v170_tokens" / "train.tokens.bin"
    train_meta = REPO / "data" / "v170_tokens" / "train.tokens.bin.meta.json"
    val_bin = REPO / "data" / "v170_tokens" / "val.tokens.bin"
    val_meta = REPO / "data" / "v170_tokens" / "val.tokens.bin.meta.json"

    for bin_path, meta_path in [(train_bin, train_meta), (val_bin, val_meta)]:
        bin_exists = bin_path.exists()
        meta_exists = meta_path.exists()

        result = {
            "bin": str(bin_path),
            "bin_exists": bin_exists,
            "bin_size": bin_path.stat().st_size if bin_exists else 0,
            "meta_exists": meta_exists,
            "consistent": False,
        }

        if bin_exists and meta_exists:
            import numpy as np

            data = np.memmap(bin_path, dtype=np.uint16, mode="r")
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            result["num_tokens"] = len(data)
            result["meta_tokens"] = meta.get("num_tokens")
            result["dtype"] = meta.get("dtype")
            result["consistent"] = len(data) == meta.get("num_tokens")
            result["vocab_size"] = meta.get("vocab_size")
            result["max_token"] = int(data.max()) if len(data) > 0 else 0

        results.append(result)

    return {"token_files": results}


def check_environment() -> Dict[str, Any]:
    """Check Python environment and dependencies."""
    import importlib

    required_packages = [
        "torch",
        "tokenizers",
        "numpy",
        "pyyaml",
        "fastapi",
        "uvicorn",
        "pydantic",
        "gradio",
        "pytest",
    ]

    results = []
    for pkg in required_packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            results.append({
                "package": pkg,
                "installed": True,
                "version": version,
            })
        except ImportError:
            results.append({
                "package": pkg,
                "installed": False,
                "version": None,
            })

    # Check GPU availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        gpu_info = []
        for i in range(device_count):
            gpu_info.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 1),
            })
    except ImportError:
        cuda_available = False
        device_count = 0
        gpu_info = []

    return {
        "python": sys.version,
        "packages": results,
        "cuda_available": cuda_available,
        "gpu_count": device_count,
        "gpus": gpu_info,
        "training_ready": cuda_available and device_count > 0,
    }


def run_all_checks() -> Dict[str, Any]:
    """Run all preflight checks."""
    print("=" * 60)
    print("TAKSH v1.9.0 Preflight Validation")
    print("=" * 60)
    print()

    # 1. File existence
    print("1. Checking required files...")
    file_results = [check_file_exists(REPO / f) for f in REQUIRED_FILES]
    missing = [f for f in file_results if not f["exists"]]
    print(f"   Found: {len(file_results) - len(missing)}/{len(file_results)}")
    if missing:
        print("   MISSING:")
        for f in missing:
            print(f"     - {f['path']}")
    print()

    # 2. Config values
    print("2. Checking config values...")
    config_results = []
    for config_path, expected in EXPECTED_CONFIG_VALUES.items():
        result = check_config_values(REPO / config_path, expected)
        config_results.append(result)
        if not result["all_match"]:
            print(f"   MISMATCH: {config_path}")
            for key, check in result.get("checks", {}).items():
                if not check["match"]:
                    print(f"     {key}: expected {check['expected']}, got {check['actual']}")
        else:
            print(f"   OK: {config_path}")
    print()

    # 3. Data integrity
    print("3. Checking data integrity...")
    data_result = check_data_integrity()
    all_consistent = all(r["consistent"] for r in data_result["token_files"])
    for r in data_result["token_files"]:
        status = "OK" if r["consistent"] else "MISMATCH"
        print(f"   {r['bin']}: {r.get('num_tokens', 'N/A')} tokens (meta: {r.get('meta_tokens', 'N/A')}) [{status}]")
    print()

    # 4. Environment
    print("4. Checking environment...")
    env_result = check_environment()
    print(f"   Python: {env_result['python'].split()[0]}")
    print(f"   CUDA available: {env_result['cuda_available']}")
    print(f"   GPU count: {env_result['gpu_count']}")
    for gpu in env_result.get("gpus", []):
        print(f"     GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']} GB)")
    missing_packages = [p["package"] for p in env_result["packages"] if not p["installed"]]
    if missing_packages:
        print(f"   Missing packages: {', '.join(missing_packages)}")
    print()

    # Summary
    all_checks_pass = (
        len(missing) == 0
        and all(r["all_match"] for r in config_results)
        and all_consistent
        and len(missing_packages) == 0
    )

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files: {'PASS' if len(missing) == 0 else 'FAIL'}")
    print(f"Configs: {'PASS' if all(r['all_match'] for r in config_results) else 'FAIL'}")
    print(f"Data: {'PASS' if all_consistent else 'FAIL'}")
    print(f"Environment: {'PASS' if len(missing_packages) == 0 else 'FAIL'}")
    print(f"GPU Available: {'YES' if env_result['cuda_available'] else 'NO'}")
    print()
    if all_checks_pass and env_result["cuda_available"]:
        print("READY FOR GPU TRAINING")
    elif all_checks_pass and not env_result["cuda_available"]:
        print("READY FOR CPU VALIDATION (GPU needed for training)")
    else:
        print("NOT READY — Fix issues above before training")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files": file_results,
        "configs": config_results,
        "data": data_result,
        "environment": env_result,
        "all_checks_pass": all_checks_pass,
        "gpu_available": env_result["cuda_available"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v1.9.0 Preflight Validation")
    parser.add_argument("--check-all", action="store_true", help="Run all checks")
    parser.add_argument("--check-configs", action="store_true", help="Check config values only")
    parser.add_argument("--check-data", action="store_true", help="Check data integrity only")
    parser.add_argument("--check-environment", action="store_true", help="Check environment only")
    args = parser.parse_args()

    if args.check_all or not any([args.check_configs, args.check_data, args.check_environment]):
        run_all_checks()
    else:
        if args.check_configs:
            for config_path, expected in EXPECTED_CONFIG_VALUES.items():
                result = check_config_values(REPO / config_path, expected)
                print(json.dumps(result, indent=2))
        if args.check_data:
            print(json.dumps(check_data_integrity(), indent=2))
        if args.check_environment:
            print(json.dumps(check_environment(), indent=2))


if __name__ == "__main__":
    main()
