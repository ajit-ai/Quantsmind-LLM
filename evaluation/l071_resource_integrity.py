"""TAKSH L0.7.1 — resource measurements and artifact integrity verification.

Measures host disk/memory (psutil/shutil; never provisioned), and verifies
the integrity of every L0.7.1 artifact plus the frozen benchmarks. Read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PINNED = {
    "evaluation/l02_python_bench_v1.json": "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6",
    "evaluation/l051_multilingual_benchmark_v1.json": "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e",
    "runs/v180_sft/checkpoint_last.pt": "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666",
    "configs/foundation/l060_foundation.yaml": "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3",
    "configs/feasibility/l051_languages.yaml": "075d4b568f052c8ebede69ccaf21cc4d0fd15a4f01df455f1431505c354ef39d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l071_resource_integrity.json")
    args = parser.parse_args(argv)

    try:
        import psutil

        vm = psutil.virtual_memory()
        memory = {"total_bytes": int(vm.total), "available_bytes": int(vm.available),
                  "percent_used": float(vm.percent), "tool": "psutil", "measured": True}
    except Exception:
        memory = {"measured": False}

    disk = shutil.disk_usage(REPO)
    disk_report = {"total_bytes": disk.total, "free_bytes": disk.free,
                   "used_bytes": disk.used, "measured": True, "tool": "shutil.disk_usage"}

    pinned = []
    for rel, expected in PINNED.items():
        path = REPO / rel
        actual = sha256_file(path) if path.is_file() else None
        pinned.append({"path": rel, "expected": expected, "actual": actual,
                       "intact": actual == expected})

    snapshot_path = REPO / "artifacts" / "l071_l04_snapshot" / "snapshot.json"
    records_path = REPO / "artifacts" / "l071_l04_snapshot" / "records_meta.jsonl"
    artifacts = []
    for path in (snapshot_path, records_path):
        artifacts.append({"path": str(path.relative_to(REPO)),
                          "exists": path.is_file(),
                          "sha256": sha256_file(path) if path.is_file() else None,
                          "bytes": path.stat().st_size if path.is_file() else None})

    tokenizer_path = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
    artifacts.append({"path": "tokenizer/taksh_code_bpe_v1.json",
                      "exists": tokenizer_path.is_file(),
                      "sha256": sha256_file(tokenizer_path),
                      "note": "working-tree CRLF copy; committed LF blob hashed separately in tests",
                      "bytes": tokenizer_path.stat().st_size})

    report = {
        "schema": "taksh.l071_resource_integrity/1",
        "generated_by": "evaluation/l071_resource_integrity.py",
        "memory": memory,
        "disk": disk_report,
        "pinned_artifacts": pinned,
        "all_pinned_intact": all(p["intact"] for p in pinned),
        "artifacts": artifacts,
        "reservations": {
            "paid_storage": "NONE", "cloud_resources": "NONE",
            "network_downloads": "NONE", "gpu": "NONE",
        },
        "integrity_policy": "read-only verification; hashes recorded before and after the milestone",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"free disk: {disk_report['free_bytes'] / 2**30:.1f} GiB; pinned intact: {report['all_pinned_intact']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
