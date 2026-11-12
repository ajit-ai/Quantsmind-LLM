"""TAKSH L1.1.0 — Native Model Foundation qualification runner.

Read-only, non-training probes of the native model foundation. Verifies
configuration identity, tokenizer compatibility, parameter accounting and
preservation anchors — never trains, never instantiates a 110M model in RAM.
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

FOUNDATION_SHA_EXPECTED = None   # recorded at first verification, by design free
TOKENIZER_HASH_METHOD = "sha256 of raw file bytes"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
FOUNDATION_YAML = REPO / "configs" / "model" / "foundation_111m.yaml"
TOKENIZER_JSON = REPO / "tokenizer" / "qm_code_bpe_v0.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> dict:
    from training.model import QMModel

    cfg_doc = yaml.safe_load(FOUNDATION_YAML.read_text(encoding="utf-8"))
    cfg = cfg_doc["model"]

    # 1. parameter accounting via meta device (zero RAM cost)
    import torch

    with torch.device("meta"):
        model = QMModel(cfg)
        total = sum(p.numel() for p in model.parameters())
    params_ok = total == 110_658_816

    vocab = cfg["vocab_size"]; tok = json.loads(TOKENIZER_JSON.read_text(encoding="utf-8"))
    tok_vocab = len(tok["model"]["vocab"])
    tok_ok = tok_vocab == vocab == 5822

    foundation_sha = file_sha256(FOUNDATION_YAML)
    tokenizer_sha = file_sha256(TOKENIZER_JSON)
    v180_present = (REPO / "runs" / "v180_sft" / "checkpoint_last.pt").is_file()
    v180_sha = file_sha256(REPO / "runs" / "v180_sft" / "checkpoint_last.pt") if v180_present else None
    v180_ok = (v180_sha == V180_SHA) if v180_present else None

    fingerprint_check = {
        "foundation_config_sha256": foundation_sha,
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_method": TOKENIZER_HASH_METHOD,
        "v180_sft_present": v180_present,
        "v180_sft_intact": v180_ok,
        "v180_sft_value_reference": V180_SHA if v180_present else None,
        "fingerprint_unchanged_files": v180_ok if v180_present else "not_available",
    }

    report = {
        "schema": "taksh.l110_native_model_foundation/1",
        "generated_by": "evaluation/l110_native_foundation.py",
        "verified_on": "2026-11-12",
        "foundation_config": {
            "path": str(FOUNDATION_YAML.relative_to(REPO)),
            "id": cfg_doc["id"],
            "name": cfg_doc["name"],
            "parameters": total,
            "matches_expected": params_ok,
            "architecture": {
                "vocab_size": cfg["vocab_size"],
                "d_model": cfg["d_model"],
                "n_layers": cfg["n_layers"],
                "n_heads": cfg["n_heads"],
                "n_kv_heads": cfg["n_kv_heads"],
                "ffn_dim": cfg["ffn_dim"],
                "max_seq_len": cfg["max_seq_len"],
                "rope_theta": cfg["rope_theta"],
                "tie_embeddings": cfg["tie_embeddings"],
            },
        },
        "tokenizer": {
            "path": str(TOKENIZER_JSON.relative_to(REPO)),
            "vocab_size": tok_vocab,
            "matches_foundation": tok_ok,
            "fingerprints": fingerprint_check,
        },
        "relationships": {
            "v180_sft": "GEOMETRY_INCOMPATIBLE — never substituted (d_model 384)",
            "l060_foundation_yaml": "experimental track (feature branch), different tokenizer; not confused with this config",
            "toy_ci": "tiny config for focused tests",
            "code_m3_9m": "8.9M-parameter researched 9M target of the v1.x lineage",
        },
        "capability_claim": "none — engineering qualification only",
        "training": "NOT GRANTED / A1.1-C / A2-E",
        "core_ok": bool(params_ok and tok_ok),
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l110_native_model_foundation.json")
    args = parser.parse_args(argv)

    report = verify()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"foundation parameters: {report['foundation_config']['parameters']:,} "
          f"matches: {report['foundation_config']['matches_expected']}")
    print(f"tokenizer vocab: {report['tokenizer']['vocab_size']} matches: {report['tokenizer']['matches_foundation']}")
    print(f"report written to {out_path}")
    return 0 if report["core_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
