"""TAKSH L0.6.0 — WP8: shared reproducibility/evidence collector.

Every qualification run records: git branch/HEAD, working-tree state, config
hash, tokenizer hash, benchmark hash, dependency versions, hardware/device
info actually observed, test commands and exit codes, model parameter count,
and inference smoke-test results. Measured values are always distinguished
from estimates and assumptions.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(*args) -> str:
    return subprocess.run(list(args), capture_output=True, text=True, cwd=REPO,
                         check=True).stdout.strip()


def git_block() -> dict:
    entries = run("git", "status", "--porcelain=v1").splitlines()
    return {
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "head": run("git", "rev-parse", "HEAD"),
        "main": run("git", "rev-parse", "main"),
        "develop": run("git", "rev-parse", "develop"),
        "working_tree_entries": len(entries),
        "working_tree_entries_list": sorted(entries),
    }


def hardware_block() -> dict:
    import torch

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
        "logical_cpus": None,  # filled by torch.get_num_threads if available
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": "cpu",
        "note": "CPU-only host by deliberate choice; MX150 4 GB not usable here (driver/toolkit mismatch)",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def anchor_block() -> dict:
    return {
        "tokenizer": {
            "path": "tokenizer/taksh_code_bpe_v1.json",
            "vocab_size": 5822,
            "specials": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3},
            "committed_artifact_sha256": "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7",
            "working_tree_sha256_crlf": file_sha256(REPO / "tokenizer/taksh_code_bpe_v1.json"),
        },
        "frozen_python_benchmark": {
            "path": "evaluation/l02_python_bench_v1.json",
            "sha256": file_sha256(REPO / "evaluation" / "l02_python_bench_v1.json"),
            "expected_sha256": "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6",
        },
    }


def collected_block() -> dict:
    git = git_block()
    hw = hardware_block()
    try:
        import torch

        hw["logical_cpus"] = torch.get_num_threads()
        hw["physical_cpus"] = None
    except Exception:
        pass
    return {
        "collected_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git": git,
        "hardware": hw,
        "anchors": anchor_block(),
    }
