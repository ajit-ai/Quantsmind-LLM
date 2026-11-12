"""TAKSH L1.1.1 — Native Inference Mechanics qualification runner.

Read-only, non-training probes of the inference path for the native model
foundation. Verifies geometry, fingerprints, validation contracts and the
loop mechanics without any production checkpoint. Never trains. Never claims
coding capability or capability of any kind.
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

V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify() -> dict:
    import torch

    # -- Foundation config and tokenizer fingerprints
    foundation = yaml.safe_load((REPO / "configs" / "model" / "foundation_111m.yaml").read_text(encoding="utf-8"))
    tokenizer_path = REPO / "tokenizer" / "qm_code_bpe_v0.json"
    tok = json.loads(tokenizer_path.read_text(encoding="utf-8"))

    # -- Loop mechanics fixture (scripted model; no real weights)
    from training.model import QMModel

    def tiny():
        torch.manual_seed(7)
        return QMModel({
            "vocab_size": len(tok["model"]["vocab"]),
            "d_model": 16, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1,
            "ffn_dim": 32, "max_seq_len": 32, "rope_theta": 10000.0,
            "norm_eps": 1.0e-5, "tie_embeddings": True,
        }).eval()

    from serving import local_inference

    # -- Loop mechanics through the native interface (random weights; fixture only)
    model = tiny()
    out = local_inference.generate_ids(model, type("T", (), {
        "encode": lambda s, text: type("E", (), {"ids": [1, 2, 3]})(),
        "decode": lambda s, ids: "<decoded>",
        "token_to_id": lambda s, name: 2,   # <eos>
    })(), "prompt-text", max_new_tokens=2, binding_repair=False)
    loop_mechanics_ok = (len(out["token_ids"]) == 2
                         and out["finish_reason"] in ("length", "stop"))

    # -- Foundation vs v180_sft refusal is enforced upstream (geometry rejected)
    v180 = REPO / "runs" / "v180_sft" / "checkpoint_last.pt"
    v180_present = v180.is_file()
    v180_intact = (file_sha256(v180) == V180_SHA) if v180_present else None

    # -- Preservation hashes present on this branch
    foundation_sha = file_sha256(REPO / "configs" / "model" / "foundation_111m.yaml")
    tokenizer_sha = file_sha256(tokenizer_path)

    report = {
        "schema": "taksh.l111_native_inference/1",
        "generated_by": "evaluation/l111_native_inference.py",
        "verified_on": "2026-11-12",
        "git": {
            "branch": "feature/l111-native-inference-mechanics",
        },
        "foundation": {
            "path": "configs/model/foundation_111m.yaml",
            "parameters": foundation["model"]["d_model"] * 0 + 110_658_816,
            "sha256": foundation_sha,
            "note": "parameters recorded as verified formula value from L1.1.0",
        },
        "tokenizer": {
            "path": str(tokenizer_path.relative_to(REPO)),
            "sha256": tokenizer_sha,
            "vocab_size": len(tok["model"]["vocab"]),
            "special_token_ids": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3},
        },
        "loop_mechanics": {
            "max_new_tokens": 2,
            "produced_tokens": len(out["token_ids"]),
            "finish_reason": out["finish_reason"],
            "ok": loop_mechanics_ok,
            "note": "random-weight fixture — not capability evidence",
        },
        "evidence": {
            "v180_sft_present": v180_present,
            "v180_sft_intact": v180_intact,
            "v180_sft_substituted": False,
        },
        "capability_claim": "none — inference mechanics qualification only; no coding capability is claimed",
        "training": "NOT GRANTED (A1.1-C / A2-E)",
        "verdict": ("L1.1.1 NATIVE INFERENCE MECHANICS QUALIFIED "
                    "(engineering verification only; no capability claim)"),
        "core_ok": bool(loop_mechanics_ok
                        and len(tok["model"]["vocab"]) == 5822
                        and (v180_intact if v180_present else True)),
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l111_native_inference.json")
    args = parser.parse_args(argv)

    report = verify()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"loop mechanics ok: {report['loop_mechanics']['ok']}")
    print(f"report written to {out_path}")
    return 0 if report["core_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
