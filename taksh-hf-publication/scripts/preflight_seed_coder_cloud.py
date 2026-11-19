"""Fail-safe preflight for Taksh Seed-Coder cloud execution (D3.1).

This command only inspects the current host and an explicitly supplied local
model directory. It does not download weights, call hosted services, upload
data, or start training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from importlib import metadata, util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from serving.providers import (
    SEED_CODER_MODEL_ID,
    SEED_CODER_REVISION,
    SEED_CODER_SOURCE,
    SeedCoderProvider,
    _seed_artifact_hashes,
)

ENVIRONMENT_CONFIG = REPO / "configs" / "runtime" / "seed-coder-cloud-environment.json"
REQUIREMENTS_FILE = REPO / "configs" / "runtime" / "seed-coder-cloud-requirements.txt"


def _check(name: str, ok: bool, detail: str, required: bool = True) -> dict:
    return {"name": name, "ok": bool(ok), "required": bool(required), "detail": detail}


def _package_version(name: str) -> str | None:
    if util.find_spec(name) is None:
        return None
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _ram_total_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def validate_artifact_hashes(
    model_path: Path,
    *,
    expected_config_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
) -> list[dict]:
    checks = []
    if not model_path.is_dir():
        return [_check("model_artifact_directory", False, f"not found: {model_path}")]
    hashes = _seed_artifact_hashes(model_path)
    checks.append(_check("model_artifact_directory", True, str(model_path)))
    checks.append(_check("config_artifact_present", hashes.get("config.json") is not None, hashes.get("config.json") or "config.json missing"))
    checks.append(_check("tokenizer_artifact_present", hashes.get("tokenizer.json") is not None, hashes.get("tokenizer.json") or "tokenizer.json missing"))
    if expected_config_sha256:
        checks.append(_check("config_sha256", hashes.get("config.json") == expected_config_sha256, f"expected {expected_config_sha256}, got {hashes.get('config.json')}"))
    if expected_tokenizer_sha256:
        checks.append(_check("tokenizer_sha256", hashes.get("tokenizer.json") == expected_tokenizer_sha256, f"expected {expected_tokenizer_sha256}, got {hashes.get('tokenizer.json')}"))
    config_path = model_path / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            architectures = config.get("architectures") or []
            model_type = config.get("model_type") or ""
            checks.append(_check(
                "model_architecture",
                "LlamaForCausalLM" in architectures or model_type == "llama",
                f"architectures={architectures!r} model_type={model_type!r}",
            ))
        except Exception as exc:
            checks.append(_check("model_architecture", False, f"config parse failed: {type(exc).__name__}: {exc}"))
    return checks


def preflight(
    *,
    model_path: Path | None = None,
    min_vram_gib: float = 16.0,
    min_ram_gib: float = 16.0,
    min_disk_gib: float = 40.0,
    expected_config_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
) -> dict:
    checks = []
    environment = json.loads(ENVIRONMENT_CONFIG.read_text(encoding="utf-8")) if ENVIRONMENT_CONFIG.exists() else {}
    checks.append(_check("environment_spec_present", ENVIRONMENT_CONFIG.is_file(), str(ENVIRONMENT_CONFIG)))
    checks.append(_check("requirements_present", REQUIREMENTS_FILE.is_file(), str(REQUIREMENTS_FILE)))
    checks.append(_check("python_version", sys.version_info >= (3, 10) and sys.version_info < (3, 15), sys.version))
    checks.append(_check("approved_model_identity", True,
                         f"{SEED_CODER_MODEL_ID} @ {SEED_CODER_REVISION} ({SEED_CODER_SOURCE})"))

    torch_version = _package_version("torch")
    checks.append(_check("torch_installed", torch_version is not None, torch_version or "torch not installed"))
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        checks.append(_check("cuda_available", cuda_available, f"cuda_available={cuda_available}, cuda_build={getattr(torch.version, 'cuda', None)}"))
        if cuda_available:
            device_count = int(torch.cuda.device_count())
            checks.append(_check("cuda_device_count", device_count > 0, f"devices={device_count}"))
            max_vram = 0.0
            for index in range(device_count):
                props = torch.cuda.get_device_properties(index)
                vram = float(props.total_memory) / 1024**3
                max_vram = max(max_vram, vram)
                checks.append(_check(f"gpu_{index}_name", True, f"{props.name} {vram:.2f} GiB"))
            checks.append(_check("gpu_vram", max_vram >= min_vram_gib, f"max_vram={max_vram:.2f} GiB, required={min_vram_gib} GiB"))
            checks.append(_check("recommended_vram", max_vram >= 24.0, f"max_vram={max_vram:.2f} GiB, recommended=24 GiB", required=False))
        else:
            checks.append(_check("gpu_vram", False, f"CUDA unavailable; required={min_vram_gib} GiB"))
    except Exception as exc:
        checks.append(_check("cuda_available", False, f"{type(exc).__name__}: {exc}"))
        checks.append(_check("gpu_vram", False, "CUDA unavailable"))

    transformers_version = _package_version("transformers")
    accelerator_version = _package_version("accelerate")
    checks.append(_check("transformers_installed", transformers_version is not None, transformers_version or "transformers not installed"))
    checks.append(_check("accelerate_installed", accelerator_version is not None, accelerator_version or "accelerate not installed"))

    ram_bytes = _ram_total_bytes()
    if ram_bytes is None:
        checks.append(_check("system_ram", False, "could not determine total RAM"))
    else:
        ram_gib = ram_bytes / 1024**3
        checks.append(_check("system_ram", ram_gib >= min_ram_gib, f"ram={ram_gib:.2f} GiB, required={min_ram_gib} GiB"))

    disk_root = str(model_path) if model_path is not None else str(REPO)
    try:
        free_gib = shutil.disk_usage(disk_root).free / 1024**3
        checks.append(_check("free_disk", free_gib >= min_disk_gib, f"free={free_gib:.2f} GiB at {disk_root}, required={min_disk_gib} GiB"))
    except Exception as exc:
        checks.append(_check("free_disk", False, f"{type(exc).__name__}: {exc}"))

    if model_path is None:
        checks.append(_check("model_artifact_directory", False, "no --model-path supplied; local artifact is required"))
    else:
        checks.extend(validate_artifact_hashes(model_path, expected_config_sha256=expected_config_sha256, expected_tokenizer_sha256=expected_tokenizer_sha256))

    provider = SeedCoderProvider(model_path)
    checks.append(_check("provider_revision", provider.identity().revision == SEED_CODER_REVISION, provider.identity().revision))
    checks.append(_check("provider_local_only", provider.capabilities().offline is True, "offline provider capability"))

    ready = all(item["ok"] for item in checks if item["required"])
    return {
        "ready": ready,
        "model": {
            "provider": "seed-coder-8b-instruct",
            "model_id": SEED_CODER_MODEL_ID,
            "revision": SEED_CODER_REVISION,
            "source": SEED_CODER_SOURCE,
        },
        "environment_config": environment,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed-Coder cloud preflight")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--min-vram-gib", type=float, default=16.0)
    parser.add_argument("--min-ram-gib", type=float, default=16.0)
    parser.add_argument("--min-disk-gib", type=float, default=40.0)
    parser.add_argument("--expected-config-sha256", default=None)
    parser.add_argument("--expected-tokenizer-sha256", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    report = preflight(
        model_path=Path(args.model_path) if args.model_path else None,
        min_vram_gib=args.min_vram_gib,
        min_ram_gib=args.min_ram_gib,
        min_disk_gib=args.min_disk_gib,
        expected_config_sha256=args.expected_config_sha256,
        expected_tokenizer_sha256=args.expected_tokenizer_sha256,
    )
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
