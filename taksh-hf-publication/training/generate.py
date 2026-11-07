"""Standalone local generation from a trained QuantsMind-LLM checkpoint.

Loads a checkpoint and its matching tokenizer from disk and generates text
autoregressively on the local machine. No hosted model is involved at any point:
the only weights used are the ones in the checkpoint you name.

What this deliberately refuses to do:

* load a checkpoint whose architecture does not match its recorded config;
* generate with a tokenizer whose vocabulary disagrees with the model;
* silently fall back to random weights if loading fails;
* report output without recording which checkpoint produced it.

Usage
-----
    python -m training.generate --ckpt runs/m3_base/checkpoint_last.pt \
        --prompt "def fibonacci(n):" --max-new-tokens 80 --temperature 0.8

    # deterministic
    python -m training.generate --ckpt ... --prompt ... --greedy

    # reproduce exactly
    python -m training.generate --ckpt ... --prompt ... --seed 7
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "quantsmind.local_generation/1"


class GenerationError(RuntimeError):
    """Generation could not run: missing, corrupt, or incompatible artifacts."""


def load_checkpoint_for_generation(ckpt_path, device="cpu"):
    """Load a checkpoint, its model config, and prove the weights match.

    Returns ``(model, info)``. Raises :class:`GenerationError` rather than
    starting from a randomly initialised model, because "it produced output"
    from random weights is exactly the failure this milestone must not hide.
    """
    import torch
    import yaml

    from training.checkpoint import CheckpointIncompatible, load_model_weights
    from training.model import QMModel

    path = Path(ckpt_path)
    if not path.exists():
        raise GenerationError(
            f"checkpoint not found: {path}\n"
            "Train one first: python -m training.pretrain --config <config>"
        )
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - corrupt or truncated file
        raise GenerationError(
            f"{path} could not be loaded ({type(exc).__name__}: {exc}). The file "
            "is corrupt or incomplete."
        ) from exc
    if not isinstance(ckpt, dict):
        raise GenerationError(
            f"{path} is not a checkpoint dictionary (got {type(ckpt).__name__})"
        )
    if ckpt.get("schema") != "quantsmind.checkpoint/1":
        raise GenerationError(
            f"{path} has schema {ckpt.get('schema')!r}, expected "
            "'quantsmind.checkpoint/1'"
        )
    state = ckpt.get("model")
    if state is None:
        raise GenerationError(f"{path} contains no 'model' weights entry")

    cfg = ckpt.get("config") or {}
    run_cfg = cfg.get("run") or {}
    # `model` means the architecture in pretraining configs, but a fine-tuning
    # config may use `model:` for policy (freeze settings). Only accept it when
    # it actually looks like an architecture.
    model_cfg = cfg.get("model") if "d_model" in (cfg.get("model") or {}) else None
    if model_cfg is None:
        candidate = cfg.get("model_cfg")
        if candidate and "d_model" in candidate:
            model_cfg = candidate
    # M4 checkpoints fine-tune an M3 base, so they record the architecture under
    # `base_model_config` rather than `model_config`.
    model_cfg_path = run_cfg.get("model_config") or run_cfg.get("base_model_config")
    if not model_cfg and model_cfg_path and Path(model_cfg_path).exists():
        loaded = yaml.safe_load(Path(model_cfg_path).read_text(encoding="utf-8"))
        model_cfg = loaded.get("model", loaded)
    if not model_cfg or "d_model" not in model_cfg:
        raise GenerationError(
            f"{path} does not record a usable model configuration (looked for "
            "run.model_config, run.base_model_config, model, model_cfg). Its "
            "architecture cannot be rebuilt, and guessing would load weights "
            "into a different model. Refusing."
        )

    model = QMModel(model_cfg)
    try:
        report = load_model_weights(model, state, source=str(path))
    except CheckpointIncompatible as exc:
        raise GenerationError(str(exc)) from exc
    if not report.ok:
        raise GenerationError(
            f"{path} did not fully load:\n{report.describe()}\n"
            "Refusing to generate from partially random weights."
        )

    model = model.to(device).eval()
    info = {
        "checkpoint": str(path),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_schema": ckpt.get("schema"),
        "tokens_processed": ckpt.get("tokens_processed"),
        "val_loss_at_checkpoint": ckpt.get("val_loss"),
        "interrupted": ckpt.get("interrupted", False),
        "model_config_source": str(model_cfg_path) if model_cfg_path else "checkpoint",
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameters_loaded": len(report.loaded),
        "max_seq_len": model_cfg.get("max_seq_len"),
        "vocab_size": model_cfg.get("vocab_size"),
        "device": device,
        "dataset_identity": ckpt.get("dataset_identity"),
        "tokenizer_identity": ckpt.get("tokenizer_identity"),
    }
    return model, info


def load_tokenizer(tokenizer_path, model_vocab):
    """Load the tokenizer and refuse a vocabulary mismatch."""
    from tokenizers import Tokenizer

    path = Path(tokenizer_path)
    if not path.exists():
        raise GenerationError(
            f"tokenizer not found: {path}\n"
            "Generation needs the tokenizer that produced the training tokens."
        )
    try:
        tok = Tokenizer.from_file(str(path))
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(
            f"could not load tokenizer {path}: {type(exc).__name__}: {exc}"
        ) from exc
    vocab = tok.get_vocab_size()
    if model_vocab and vocab != model_vocab:
        raise GenerationError(
            f"tokenizer {path} has vocab {vocab} but the model was built for "
            f"{model_vocab}. Token ids would not correspond to what the model "
            "learned. Pass the tokenizer recorded in the checkpoint."
        )
    return tok, vocab


def generate_text(model, tokenizer, prompt, max_new_tokens=80, temperature=0.8,
                  top_k=50, top_p=0.95, greedy=False, stop=None, seed=None,
                  device="cpu"):
    """Autoregressive generation with temperature, top-k and nucleus filtering."""
    import torch
    import torch.nn.functional as F

    if seed is not None:
        torch.manual_seed(seed)

    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise GenerationError(
            "the prompt encoded to zero tokens; the tokenizer cannot represent it"
        )
    max_seq_len = int(model.cfg.get("max_seq_len") or 1024)
    if len(prompt_ids) > max_seq_len:
        raise GenerationError(
            f"prompt is {len(prompt_ids)} tokens but the model context is "
            f"{max_seq_len}. Shorten the prompt; this model does not silently "
            "truncate input."
        )

    eos_id = tokenizer.token_to_id("<eos>")
    stop_list = [stop] if isinstance(stop, str) else list(stop or [])
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    finish_reason = "length"

    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = x if x.size(1) <= max_seq_len else x[:, -max_seq_len:]
            logits, _ = model(window)
            next_logits = logits[:, -1, :]
            if greedy or not temperature or temperature <= 0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / max(temperature, 1e-5)
                if top_k and top_k > 0:
                    k = min(top_k, scaled.size(-1))
                    threshold = torch.topk(scaled, k)[0][:, [-1]]
                    scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                if top_p is not None and 0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = (cumulative - sorted_probs) > top_p
                    remove[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    # Scatter along the vocabulary axis. probs is (batch, vocab),
                    # so dim 0 is the batch axis and scatter_(0, ...) raises
                    # "index out of bounds for dimension 0 with size 1".
                    probs = torch.zeros_like(probs).scatter_(
                        -1, sorted_idx, sorted_probs
                    )
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_id = torch.multinomial(probs, num_samples=1)
            token = int(next_id.item())
            if eos_id is not None and token == eos_id:
                finish_reason = "stop"
                break
            generated.append(token)
            x = torch.cat([x, next_id], dim=1)
            if stop_list:
                tail = tokenizer.decode(generated)
                hit = next((s for s in stop_list if s and s in tail), None)
                if hit is not None:
                    generated = tokenizer.encode(tail[: tail.index(hit)]).ids
                    finish_reason = "stop"
                    break

    return {
        "prompt": prompt,
        "completion": tokenizer.decode(generated),
        "token_ids": generated,
        "finish_reason": finish_reason,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(generated),
        "eos_id": eos_id,
        "full_text": tokenizer.decode(prompt_ids + generated),
    }


def run_generation(ckpt, prompt, tokenizer_path=None, max_new_tokens=80,
                   temperature=0.8, top_k=50, top_p=0.95, greedy=False,
                   stop=None, seed=None, device="cpu"):
    """Load, generate, and return a record naming exactly what was used."""
    model, info = load_checkpoint_for_generation(ckpt, device=device)
    tokenizer_path = tokenizer_path or (
        (info.get("tokenizer_identity") or {}).get("path")
    )
    if not tokenizer_path:
        raise GenerationError(
            "the checkpoint records no tokenizer and none was given. Pass "
            "--tokenizer with the file used to produce the training tokens."
        )
    tok, vocab = load_tokenizer(tokenizer_path, info["vocab_size"])
    result = generate_text(
        model, tok, prompt, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy,
        stop=stop, seed=seed, device=device,
    )
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "checkpoint": info,
        "tokenizer": {"path": str(tokenizer_path), "vocab_size": vocab},
        "sampling": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "greedy": greedy,
            "seed": seed,
            "stop": stop,
        },
        "result": result,
        "provenance": (
            "Text generated locally from the named checkpoint using the named "
            "tokenizer. No hosted or external language model was used."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate text locally from a QuantsMind-LLM checkpoint"
    )
    parser.add_argument("--ckpt", required=True, help="checkpoint to load")
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--tokenizer", default=None,
        help="defaults to the tokenizer recorded in the checkpoint",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true", help="argmax decoding")
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--record", default=None,
        help="write the generation record (including the output) as JSON",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args(argv)

    try:
        record = run_generation(
            args.ckpt, args.prompt, tokenizer_path=args.tokenizer,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, greedy=args.greedy,
            stop=args.stop, seed=args.seed, device=args.device,
        )
    except GenerationError as exc:
        print(f"generation failed: {exc}")
        return 1

    if args.record:
        path = Path(args.record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(record, indent=2))
        return 0

    info = record["checkpoint"]
    print(f"checkpoint : {info['checkpoint']}")
    print(f"  step        : {info['checkpoint_step']}")
    print(f"  parameters  : {info['parameters']:,} "
          f"({info['parameters_loaded']} tensors loaded)")
    print(f"  tokens seen : {info['tokens_processed']}")
    if info.get("val_loss_at_checkpoint") is not None:
        print(f"  val loss    : {info['val_loss_at_checkpoint']:.4f}")
    if info.get("interrupted"):
        print("  NOTE: this checkpoint came from an interrupted run")
    print(f"tokenizer  : {record['tokenizer']['path']} "
          f"(vocab {record['tokenizer']['vocab_size']})")
    print(f"device     : {info['device']}")
    sampling = record["sampling"]
    print(f"sampling   : "
          + (f"greedy" if sampling["greedy"] else
             f"temperature={sampling['temperature']} top_k={sampling['top_k']} "
             f"top_p={sampling['top_p']}")
          + (f" seed={sampling['seed']}" if sampling["seed"] is not None else ""))
    print()
    print("--- PROMPT ---")
    print(record["result"]["prompt"])
    print("--- COMPLETION (unedited model output) ---")
    print(record["result"]["completion"])
    print("--- END ---")
    result = record["result"]
    print(f"prompt_tokens={result['prompt_tokens']} "
          f"completion_tokens={result['completion_tokens']} "
          f"finish_reason={result['finish_reason']}")
    print(f"provenance: {record['provenance']}")
    if args.record:
        print(f"record written: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())