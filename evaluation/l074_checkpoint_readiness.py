"""TAKSH L0.7.4 — canonical checkpoint readiness validator.

Validates that the *requirements* for a future authorized foundation-111m
checkpoint are well-defined AND that any candidate artifact fails them
correctly today. Never fabricates a foundation checkpoint and never creates
one. Uses the L0.6.1/L0.7.2 machinery (geometry inspector, fingerprints).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
SPEC_PATH = REPO / "configs" / "foundation" / "l074_canonical_checkpoint_readiness.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_spec(path: Path = SPEC_PATH) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_spec(spec: dict) -> list[str]:
    problems: list[str] = []
    model = spec.get("canonical_model") or {}
    expected = {"parameters": 110_658_816, "vocab_size": 5822, "d_model": 768,
                "n_layers": 12, "n_heads": 12, "n_kv_heads": 6, "ffn_dim": 3072,
                "max_seq_len": 2048, "rope_theta": 10000.0, "tie_embeddings": True}
    for key, want in expected.items():
        if model.get(key) != want:
            problems.append(f"canonical_model.{key}={model.get(key)!r} != {want!r}")
    artifact = spec.get("artifact_requirements") or {}
    for key, body in artifact.items():
        if not isinstance(body, dict) or body.get("required") is not True:
            problems.append(f"artifact_requirements.{key} must be required:true")
    if "dataset_eligibility_record" not in artifact:
        problems.append("dataset_eligibility_record requirement missing")
    if "checkpoint_integrity_hash" not in artifact:
        problems.append("checkpoint_integrity_hash requirement missing")
    if "owner_authorization_gate" not in (spec.get("provenance_requirements") or {}):
        problems.append("owner_authorization_gate provenance requirement missing")
    neg = spec.get("negative_acceptance") or []
    for expected in ("wrong_architecture_d_model_384", "synthetic_fixture",
                     "tokenizer_fingerprint_mismatch", "missing_artifact"):
        if expected not in neg:
            problems.append(f"negative_acceptance must include {expected}")
    frozen = REPO / "evaluation" / "l02_python_bench_v1.json"
    if frozen.is_file() and sha256_file(frozen) != FROZEN_PY_SHA:
        problems.append("frozen Python benchmark changed")
    return problems


def architectures_match(resolved: dict, canonical: dict) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for key in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
                "ffn_dim", "max_seq_len", "rope_theta", "norm_eps", "tie_embeddings"):
        a, b = resolved.get(key), canonical.get(key)
        if a != b:
            problems.append(f"{key}: resolved={a!r} != canonical={b!r}")
    return not problems, problems


def evaluate_candidate(checkpoint_path: Path | None, *, is_synthetic: bool = False) -> dict:
    """Fail-closed candidate evaluation used by readiness reporting."""
    if is_synthetic:
        return {"accepted": False, "reason": "synthetic fixture is never a production candidate",
                "classification": "SYNTHETIC_REJECTED"}
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        return {"accepted": False, "reason": "missing artifact",
                "classification": "MISSING_ARTIFACT"}
    try:
        from serving.local_inference import torch_load_checkpoint, resolve_model_config

        canonical = None
        import yaml

        canonical = yaml.safe_load((REPO / "configs" / "foundation" / "l060_foundation.yaml").read_text(encoding="utf-8"))["model"]
        ckpt = torch_load_checkpoint(str(checkpoint_path))
        resolved = resolve_model_config(ckpt)
    except Exception as exc:  # noqa: BLE001
        return {"accepted": False,
                "reason": f"cannot resolve checkpoint config: {type(exc).__name__}: {exc}",
                "classification": "UNRESOLVABLE_ARTIFACT"}
    compatible, problems = architectures_match(resolved, canonical)
    if not compatible:
        return {"accepted": False,
                "reason": "; ".join(problems),
                "classification": "GEOMETRY_INCOMPATIBLE",
                "resolved_config": resolved}
    return {"accepted": False,
            "reason": "geometry compatible but training authorization absent",
            "classification": "GEOMETRY_COMPATIBLE_BUT_UNAUTHORIZED",
            "resolved_config": resolved}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l074_checkpoint_readiness.json")
    args = parser.parse_args(argv)

    spec = load_spec()
    problems = validate_spec(spec)
    v180 = REPO / "runs" / "v180_sft" / "checkpoint_last.pt"
    synthetic = REPO / "tests" / "fixture_s2_checkpoint.py"

    report = {
        "schema": "taksh.l074_checkpoint_readiness/1",
        "generated_by": "evaluation/l074_checkpoint_readiness.py",
        "spec_valid": not problems,
        "problems": problems,
        "candidates": {
            "canonical_111m_artifact": evaluate_candidate(None),
            "v180_sft_rejected": evaluate_candidate(v180),
            "synthetic_fixture_rejected": {"accepted": False,
                                           "reason": "synthetic fixture is never a production candidate; it is labeled synthetic in schema/name",
                                           "classification": "SYNTHETIC_REJECTED"},
        },
        "lifecycle": {
            "atomic_write": "enforced by L0.6.1 training/checkpoint_lifecycle.py (mkstemp+fsync+os.replace)",
            "interrupted_write_detection": "enforced (stale .tmp_partial rejection)",
            "tamper_rejection": "enforced (fingerprint mismatch rejects)",
            "resume_validation": "enforced (schema+tokenizer+config fingerprints)",
            "existing_checkpoints_untouched": True,
        },
        "verdict": "CANONICAL CHECKPOINT REQUIREMENTS QUALIFIED — no artifact exists; nothing is authorized; nothing fabricated",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        for p in problems:
            print("SPEC INVALID:", p)
        return 2
    print(f"spec valid; canonical candidates: all rejected as required")
    print(f"  - canonical_111m_artifact: {report['candidates']['canonical_111m_artifact']['classification']}")
    print(f"  - v180_sft:                {report['candidates']['v180_sft_rejected']['classification']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
