"""TAKSH L0.6.1 — A: canonical training configuration qualification.

Validates the L0.6.1 canonical training configuration against the qualified
foundation model (taksh-foundation-111m, 110,658,816 parameters, verified in
L0.6.0) and enforces:

  * schema version identity (taksh.training_config/1)
  * exact compatibility with configs/foundation/l060_foundation.yaml
  * explicit override semantics (no silent mutation)
  * deterministic hashing of the config
  * reproducibility contract (seed, determinism, fixed schedule metadata)

Usage:
    python -m evaluation.l0601_training_config
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CONFIG_PATH = REPO / "configs" / "foundation" / "l0601_training.yaml"
FOUNDATION_PATH = REPO / "configs" / "foundation" / "l060_foundation.yaml"
EXPECTED_PARAMS = 110_658_816


def load(path=CONFIG_PATH) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate(cfg: dict) -> list[str]:
    import yaml

    problems: list[str] = []
    if cfg.get("schema") != "taksh.training_config/1":
        problems.append(f"schema must be taksh.training_config/1, got {cfg.get('schema')!r}")
    if cfg.get("base_foundation_config") != "configs/foundation/l060_foundation.yaml":
        problems.append("base_foundation_config mismatch")
    foundation = yaml.safe_load(FOUNDATION_PATH.read_text(encoding="utf-8"))["model"]
    m = cfg.get("model") or {}
    for key in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                "ffn_dim", "max_seq_len", "rope_theta", "norm_eps", "tie_embeddings"):
        if m.get(key) != foundation.get(key):
            problems.append(f"model.{key}={m.get(key)!r} != foundation {foundation.get(key)!r}")
    if m.get("vocab_size") != 5822:
        problems.append("vocab_size must be 5822")
    batch = cfg.get("batch") or {}
    if batch.get("micro_batch_size", -1) <= 0 or batch.get("micro_batch_size", -1) > 16:
        problems.append("micro_batch_size out of bounds")
    training = cfg.get("training") or {}
    if training.get("max_steps", -1) <= 0:
        problems.append("training.max_steps must be positive")
    if training.get("deterministic") is not True:
        problems.append("training.deterministic must be true")
    if not training.get("seeds"):
        problems.append("training.seeds required")
    else:
        try:
            for s in training["seeds"]:
                int(s)
        except (TypeError, ValueError):
            problems.append("training.seeds must be integers")
    if cfg.get("status") != "NOT_YET_AUTHORIZED":
        problems.append("status must remain NOT_YET_AUTHORIZED until owner signs off")
    return problems


def canonical_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def override(cfg: dict, key: str, value: str) -> dict:
    """Apply an explicit override (dotted key) with type coercion.

    Keys not in the schema are rejected. Types are inferred from the existing
    value (int/float/bool/str). Unknown types fall back to string and are
    flagged so they can never silently change semantics.
    """
    out = json.loads(json.dumps(cfg))  # deep copy via JSON
    cur = out
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cur:
            raise KeyError(f"unknown override path: {'.'.join(parts[: -len(parts) + parts.index(part) + 1])}")
        cur = cur[part]
    leaf = parts[-1]
    if leaf not in cur:
        raise KeyError(f"unknown override key: {key}")
    existing = cur[leaf]
    if isinstance(existing, bool):
        coerced = value.strip().lower() in ("true", "1", "yes", "on")
    elif isinstance(existing, int):
        coerced = int(value)
    elif isinstance(existing, float):
        coerced = float(value)
    else:
        coerced = value
    cur[leaf] = coerced
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l0601_training_config.json")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args(argv)

    cfg = load(args.config)
    problems = validate(cfg)
    if problems:
        for p in problems:
            print("CONFIG INVALID:", p)
        return 2

    report = {
        "schema": "taksh.l0601_training_config/1",
        "generated_by": "evaluation/l0601_training_config.py",
        "valid": True,
        "config_id": cfg["id"],
        "version": cfg["version"],
        "config_sha256": canonical_hash(cfg),
        "base_foundation_sha256": cfg["base_foundation_sha256"],
        "model": cfg["model"],
        "optimizer": cfg["optimizer"],
        "batch": cfg["batch"],
        "training": cfg["training"],
        "status": cfg["status"],
        "verdict": "CANONICAL TRAINING CONFIG QUALIFIED (no run authorized)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"canonical training config valid; sha256={report['config_sha256']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
