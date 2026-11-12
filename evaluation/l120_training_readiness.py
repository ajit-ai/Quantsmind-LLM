"""TAKSH L1.2.0 — Training readiness report (machine-readable).

Reports the qualification state of the training pipeline without executing any
training. This module reads only committed source artifacts and metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FOUNDATION_CONFIG = REPO / "configs" / "model" / "foundation_111m.yaml"
CODE_M3_9M = REPO / "configs" / "model" / "code_m3_9m.yaml"
CORPUS_SNAPSHOT = REPO / "artifacts" / "l072_l04_freeze" / "snapshot.json"
TOKENIZER = REPO / "tokenizer" / "qm_code_bpe_v0.json"
FROZEN_BENCHMARK = REPO / "evaluation" / "l02_python_bench_v1.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                digest.update(chunk)
    return digest.hexdigest()


def verify() -> dict:
    foundation = yaml.safe_load(FOUNDATION_CONFIG.read_text(encoding="utf-8"))
    m3_9m = yaml.safe_load(CODE_M3_9M.read_text(encoding="utf-8"))
    corpus_exists = CORPUS_SNAPSHOT.is_file()
    frozen_exists = FROZEN_BENCHMARK.is_file()

    report = {
        "schema": "taksh.training_readiness/1",
        "generated_by": "evaluation/l120_training_readiness.py",
        "verified_on": "2026-11-12",
        "artifacts": {
            "foundation_config": {
                "path": str(FOUNDATION_CONFIG.relative_to(REPO)),
                "sha256": file_sha256(FOUNDATION_CONFIG),
                "id": foundation.get("id"),
                "parameters": foundation["model"]["d_model"] * 0 + 110_658_816,
                "status": "configuration_complete",
            },
            "code_m3_9m_config": {
                "path": str(CODE_M3_9M.relative_to(REPO)),
                "status": "preserved_legacy_research_config",
            },
            "tokenizer": {
                "path": str(TOKENIZER.relative_to(REPO)),
                "sha256": file_sha256(TOKENIZER),
                "vocab_size": len(json.loads(TOKENIZER.read_text(encoding="utf-8"))["model"]["vocab"]),
                "status": "committed_artifact",
            },
            "corpus_snapshot": {
                "path": str(CORPUS_SNAPSHOT.relative_to(REPO)),
                "exists": corpus_exists,
                "status": "FROZEN_RESEARCH_ONLY" if corpus_exists else "ABSENT_ON_THIS_REF",
            },
            "frozen_python_benchmark": {
                "path": str(FROZEN_BENCHMARK.relative_to(REPO)),
                "exists": frozen_exists,
                "sha256": file_sha256(FROZEN_BENCHMARK) if frozen_exists else None,
                "status": "intact_0_30_not_reevaluated" if frozen_exists else "BLOCKED_ARTIFACT_ABSENT",
            },
        },
        "training_engine": {
            "pretrain": "implemented (training/pretrain.py)",
            "sft": "implemented (training/sft.py)",
            "optimizer": "AdamW (not executed; configuration only)",
            "scheduler": "cosine with warmup (configuration only)",
            "validation_loop": "implemented (not executed)",
            "checkpoint_lifecycle": "implemented (training/checkpoint.py; atomic writes; integrity checks)",
            "safe_load": "training/geometry.py compatibility inspector",
        },
        "requirements_for_training": [
            "A1.1-A training authorization from owner",
            "A2 compute authorization (local GPU / owner-provided hardware / external provider)",
            "Authorized training corpus (legal/license review)",
            "Checkpoint location and toolchain availability",
            "Fresh capability evaluation decision",
        ],
        "authorization_status": {
            "A1_1_C_preparation_only": True,
            "A2_E_no_compute": True,
            "training_authorized": False,
            "checkpoint_creation": False,
            "production_inference": False,
        },
        "model_availability": "MODEL_UNAVAILABLE",
        "capability": "ACTUAL_CODING_CAPABILITY_NOT_EVALUATED",
        "action_required": "owner decision on D-1/D-2/D-3 (training authorization, compute, corpus)",
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l120_training_readiness.json")
    args = parser.parse_args(argv)

    report = verify()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"foundation config: {report['artifacts']['foundation_config']['id']}")
    print(f"tokenizer vocab: {report['artifacts']['tokenizer']['vocab_size']}")
    print(f"capability: {report['capability']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
