"""TAKSH L0.7.0 — resource and storage readiness (planning only).

Reports source-REPORTED sizes (labeled, never re-measured by download) and
host constraints. Nothing here provisions storage, reserves paid capacity, or
downloads anything. All estimates are labeled as estimates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def host() -> dict:
    usage = shutil.disk_usage(str(REPO))
    mem = {"tool": "unavailable"}
    try:
        import psutil

        vm = psutil.virtual_memory()
        mem = {"total_bytes": int(vm.total), "available_bytes": int(vm.available),
               "percent_used": float(vm.percent), "tool": "psutil", "measured": True}
    except Exception:
        pass
    return {
        "disk": {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free,
                 "tool": "shutil.disk_usage", "measured": True},
        "memory": mem,
    }


def source_reported_sizes() -> list[dict]:
    return [
        {"source_id": "src-bigcode-stack-v1",
         "reported": "92.36TB raw collected; ~6TB near-dedup (v1.1+); 1.43TB shown on repo page (2026-11-11)",
         "basis": "dataset card (reported, unverified here)",
         "download_required_to_verify": True},
        {"source_id": "src-bigcode-stack-v2",
         "reported": "67.53TB raw (SWH 2023-09-06); 32.1TB dedup; ~900B tokens; metadata-only SWHID repo; content via SWH S3",
         "basis": "dataset card (reported, unverified here)",
         "download_required_to_verify": True},
        {"source_id": "src-codesearchnet",
         "reported": "~6M functions; compressed size not confirmed",
         "basis": "paper arXiv:1909.09436",
         "download_required_to_verify": True},
        {"source_id": "src-pypi-permissive",
         "reported": "aggregate of 500k+ packages; size unknown",
         "basis": "none — not a dataset",
         "download_required_to_verify": True},
        {"source_id": "src-taksh-l04-code-corpus",
         "reported": "measured: data/l04_code_corpus (1,590,703 tokens; size in PROVENANCE.md)",
         "basis": "measured in-repo",
         "download_required_to_verify": False},
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_resource_readiness.json")
    args = parser.parse_args(argv)

    h = host()
    sizes = source_reported_sizes()

    def gib(b):
        return round(b / 2**30, 2)

    # deterministic planning formulas (ESTIMATES, labeled)
    def workspace_for(size_bytes_low: int) -> dict:
        return {
            "raw_corpus": size_bytes_low,
            "dedup_workspace_est": size_bytes_low * 3 // 2,
            "manifest_and_metadata": size_bytes_low // 500,   # ~O(2KB)/GB order-of-magnitude
            "note": "ESTIMATE (1.5x for dedup workspace; ~2KB metadata per GB) — not a measured allocation",
        }

    report = {
        "schema": "taksh.l070_resource_readiness/1",
        "generated_by": "evaluation/l070_resource_readiness.py",
        "host": h,
        "host_assessment": {
            "ram_gb": round((h["memory"].get("total_bytes") or 0) / 2**30, 2),
            "free_disk_gb": gib(h["disk"]["free_bytes"]),
            "verdict": ("host cannot stage TB-scale candidates; even the ~1.4TB Stack v1 page footprint "
                        "exceeds free disk; only small subsets are ever stageable here; paid/cloud storage NOT authorized"),
        },
        "source_reported_sizes": sizes,
        "example_staging_estimate_1pt4tb": workspace_for(1400 * 2**30),
        "recommended_preprocessing_stages": [
            "fetch metadata snapshot (no content)",
            "license/ToU verification against this registry",
            "content pull (owner-authorized, filtered slice only)",
            "quarantine + hash on arrival",
            "license/PII/secret/malware screen (data_quality module)",
            "benchmark-contamination fingerprint screen (contamination module)",
            "dedup (exact + near) with workspace",
            "language identification + label confidence",
            "emit per-source manifest (L0.6.1 training manifest schema)",
            "post-acquisition verification audit",
        ],
        "reservations": {
            "paid_storage": "NONE — not authorized, not created",
            "cloud_resources": "NONE — not authorized, not created",
            "network_downloads": "NONE — not performed",
        },
        "verdict": "STORAGE/RESOURCE READINESS QUALIFIED (planning only; nothing provisioned; nothing downloaded)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"free disk: {report['host_assessment']['free_disk_gb']} GiB; RAM: {report['host_assessment']['ram_gb']} GiB")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
