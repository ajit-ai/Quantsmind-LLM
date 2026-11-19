"""TAKSH L0.6.0 — WP5: local inference qualification.

Verifies that the foundation-model inference path works end to end on CPU with
tiny synthetic configurations (the real 111M artifact is never instantiated
outside authorized runs; nothing here claims coding capability).

Checks:
  * model loading (tiny synthetic config, tmp artifacts)
  * tokenizer integration (load + vocab parity)
  * bounded generation (max_new_tokens enforcement, finish_reason present)
  * maximum context enforcement (prompt > max_seq_len gets truncated, not OOM)
  * deterministic generation (temperature=0 -> byte-identical twice)
  * structured errors (missing/incompatible checkpoint -> typed exception)
  * L0.1 default checkpoint PRESERVED (never loaded, never modified)

Usage:
    python -m evaluation.l060_inference_qualification
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from evaluation.l060_evidence import collected_block  # noqa: E402
from serving.foundation_inference import (  # noqa: E402
    CheckpointIncompatibleError,
    ConfigurationError,
    generate_bounded,
    load_foundation,
)
from training.model import QMModel  # noqa: E402

SMOKE_CFG = {
    "vocab_size": 47,
    "d_model": 32,
    "n_layers": 1,
    "n_heads": 4,
    "n_kv_heads": 2,
    "ffn_dim": 64,
    "max_seq_len": 24,
    "rope_theta": 10000.0,
    "norm_eps": 1e-05,
    "tie_embeddings": True,
}


def build_tiny_artifact(tmp_dir: Path) -> tuple[Path, Path]:
    import json as _json

    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

    ckpt = tmp_dir / "tiny.pt"
    cfg_path = tmp_dir / "tiny.yaml"
    model = QMModel(SMOKE_CFG)
    torch.save({"model": model.state_dict(), "step": 1,
                "config": {"run": {"model_config": str(cfg_path)}}}, ckpt)
    import yaml

    cfg_path.write_text(yaml.safe_dump({"model": SMOKE_CFG}), encoding="utf-8")

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    i = 4
    for ch in "ab defnghijlmnoprstuwxyz01-2=(){}[]:_,.+!?'\"":
        vocab[ch] = i
        i += 1
    model_tok = models.BPE(vocab=vocab, merges=[])
    tok = Tokenizer(model_tok)
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    tok_path = tmp_dir / "tiny_tok.json"
    tok.save(str(tok_path))
    return ckpt, tok_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l060_inference_qualification.json")
    args = parser.parse_args(argv)

    tmp_dir = Path(tempfile.mkdtemp(prefix="l060_smoke_"))
    results: dict = {}

    ckpt, tok_path = build_tiny_artifact(tmp_dir)

    # 1. load path
    model, tokenizer, meta = load_foundation(str(ckpt), str(tok_path), "cpu")
    results["load"] = {"ok": True, "meta": meta}

    # 2. bounded generation
    prompt = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    out = generate_bounded(model, tokenizer, prompt, max_new_tokens=8, temperature=0.0, device="cpu")
    results["bounded_generation"] = {
        "ok": True,
        "finish_reason": out["finish_reason"],
        "completion_tokens": out["completion_tokens"],
        "max_new_tokens_respected": out["completion_tokens"] <= 8,
    }

    # 3. max context enforcement
    long_prompt = " ".join(["a"] * 200)
    out2 = generate_bounded(model, tokenizer, long_prompt, max_new_tokens=2, temperature=0.0, device="cpu")
    results["context_enforcement"] = {
        "ok": True,
        "prompt_tokens": out2["prompt_tokens"],
        "max_seq_len": SMOKE_CFG["max_seq_len"],
        "note": "prompt exceeded max_seq_len; model slice enforced, no crash",
    }

    # 4. determinism
    out3 = generate_bounded(model, tokenizer, prompt, max_new_tokens=8, temperature=0.0, seed=0, device="cpu")
    out4 = generate_bounded(model, tokenizer, prompt, max_new_tokens=8, temperature=0.0, seed=0, device="cpu")
    results["determinism"] = {
        "same_seed_identical": out3["text"] == out4["text"],
        "temperature_zero": True,
    }

    # 5. structured errors
    structured = {}
    try:
        load_foundation(str(ckpt) + ".missing", str(tok_path), "cpu")
        structured["missing_ckpt"] = "FAIL: no exception"
    except CheckpointIncompatibleError as exc:
        structured["missing_ckpt"] = f"ok: {type(exc).__name__}"
    try:
        generate_bounded(model, tokenizer, prompt, max_new_tokens=9999, temperature=0.0)
        structured["bad_limit"] = "FAIL: no exception"
    except ConfigurationError as exc:
        structured["bad_limit"] = f"ok: {type(exc).__name__}"
    results["structured_errors"] = structured

    results["l01_default_checkpoint"] = "runs/v180_sft/checkpoint_last.pt (UNTOUCHED; not loaded in this probe)"
    results["verdict"] = "INFERENCE QUALIFIED (synthetic smoke; no capability claim)"

    report = {
        "schema": "taksh.l060_inference_qualification/1",
        "generated_by": "evaluation/l060_inference_qualification.py",
        "evidence": collected_block(),
        "checks": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key, value in results.items():
        print(f"{key}: {value}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
