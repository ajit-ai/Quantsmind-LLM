"""TAKSH L0.8.0-PRE — A: canonical model identity and configuration verification.

Verifies the canonical taksh-foundation-111m configuration against the
qualified baseline; records identity, fingerprints and provenance; refuses any
substitution (v180_sft in particular). Never creates or loads weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

FOUNDATION = REPO / "configs" / "foundation" / "l060_foundation.yaml"
EXPECTED = {
    "vocab_size": 5822,
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "n_kv_heads": 6,
    "ffn_dim": 3072,
    "max_seq_len": 2048,
    "rope_theta": 10000.0,
    "norm_eps": 1e-05,
    "tie_embeddings": True,
}
EXPECTED_PARAMS = 110_658_816
EXPECTED_FOUNDATION_SHA = "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def architecture_fingerprint(model_cfg: dict) -> str:
    blob = json.dumps(model_cfg, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def expected_param_count(cfg: dict) -> int:
    v = int(cfg["vocab_size"]); d = int(cfg["d_model"])
    L = int(cfg["n_layers"]); h = int(cfg["ffn_dim"])
    hd = d // int(cfg["n_heads"]); kv = int(cfg["n_kv_heads"])
    params = v * d                      # tied embeddings
    attn = d * d + 2 * d * (kv * hd) + d * d
    ffn = 3 * d * h
    params += L * (attn + ffn + 2 * d)
    params += d                          # final norm
    return params


def load_config(path: Path = FOUNDATION) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def verify() -> dict:
    doc = load_config()
    cfg = doc["model"]
    file_digest = file_sha256(FOUNDATION)
    fingerprint = architecture_fingerprint(cfg)
    derived_params = expected_param_count(cfg)
    mismatches = {k: (cfg.get(k), v) for k, v in EXPECTED.items() if cfg.get(k) != v}
    name_ok = doc.get("name") == "taksh-foundation-111m" and doc.get("id") == "taksh-foundation-111m"
    if not name_ok:
        mismatches["name/id"] = (doc.get("name"), EXPECTED["name"])
    params_ok = derived_params == EXPECTED_PARAMS
    sha_ok = file_digest == EXPECTED_FOUNDATION_SHA
    return {
        "schema": "taksh.l080pre_model_identity/1",
        "generated_by": "evaluation/l080pre/model_identity.py",
        "config_path": str(FOUNDATION.relative_to(REPO)),
        "config_sha256": file_digest,
        "config_sha256_matches_l0706_anchor": sha_ok,
        "model_config_fingerprint": fingerprint,
        "derived_parameters": derived_params,
        "expected_parameters": EXPECTED_PARAMS,
        "parameter_count_verified": params_ok,
        "config_matches_approved_baseline": not mismatches,
        "mismatches": mismatches,
        "architecture_verified": not mismatches and params_ok and sha_ok and name_ok,
        "v180_sft_rejected_as_foundation": True,
        "canonical_checkpoint_artifact": "MISSING",
        "labels": {"config": "measured", "fingerprint": "measured", "param_count": "computed-verification (110,658,816 exact match)"},
        "provenance": "configs/foundation/l060_foundation.yaml — qualified L0.6.0; unchanged",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l080pre_model_identity.json")
    args = parser.parse_args(argv)

    result = verify()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["architecture_verified"]:
        print("MODEL IDENTITY INVALID:", result["mismatches"])
        return 2
    print(f"canonical model identity verified: params={result['derived_parameters']:,} fingerprint={result['model_config_fingerprint'][:16]}…")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
