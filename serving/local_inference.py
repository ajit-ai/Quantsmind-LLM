"""Local inference server — loads your own trained weights, generates text, serves Gradio UI.

Usage:
  python -m serving.local_inference --ckpt runs/local_15m/checkpoint_last.pt

Requires: trained checkpoint from training/pretrain.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taksh.kv_cache import KVCache  # noqa: E402


def torch_load_checkpoint(path: str, *, allow_unsafe_trusted: bool = False):
    """Load a checkpoint, safe by default.

    ``torch.load`` executes pickle opcodes, so a checkpoint is executable code by
    construction. Safe loading (``weights_only=True``) is the default and the
    only implicit mode. ``weights_only=False`` is reachable only via the
    explicit trusted-artifact boundary: the caller must pass
    ``allow_unsafe_trusted=True`` and the path must come from operator startup
    configuration, never from a request. There is no implicit fallback.
    """
    if not allow_unsafe_trusted:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - refuse, never fall back
            raise RuntimeError(
                f"Refusing to load checkpoint {path} with weights_only=True "
                f"({type(exc).__name__}: {exc}). This loader defaults to safe "
                "deserialisation; unsafe loading requires the explicit "
                "trusted-artifact boundary (allow_unsafe_trusted=True from "
                "operator startup configuration only)."
            ) from exc
    print(
        f"WARNING: loading {path} with weights_only=False (pickle execution). "
        "This requires explicit trusted-artifact authorization and must only "
        "be used with operator-supplied startup paths.",
        file=sys.stderr,
    )
    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_model_config(ckpt):
    import yaml

    config = ckpt.get("config") or {}
    run_cfg = config.get("run") or {}
    config_path = run_cfg.get("model_config") or run_cfg.get("base_model_config")
    if config_path:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model config recorded in checkpoint not found: {config_path}"
            )
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        model_cfg = loaded.get("model", loaded)
    else:
        model_cfg = ckpt.get("model_cfg") or config.get("model") or config
        if "model" in model_cfg:
            model_cfg = model_cfg["model"]

    if not model_cfg or "d_model" not in model_cfg:
        raise ValueError(
            "Checkpoint does not record a usable model configuration; refusing "
            "to load weights into a guessed architecture."
        )
    return model_cfg


def load_model(checkpoint_path: str, device: str, *, allow_unsafe_trusted: bool = False):
    from training.checkpoint import load_model_weights
    from training.model import QMModel

    ckpt = torch_load_checkpoint(checkpoint_path, allow_unsafe_trusted=allow_unsafe_trusted)
    model_cfg = resolve_model_config(ckpt)

    model = QMModel(model_cfg)
    report = load_model_weights(model, ckpt["model"], source=str(checkpoint_path))
    step = ckpt.get("step", "?")
    print(f"loaded checkpoint step={step} params={model.num_params():,} "
          f"tensors={len(report.loaded)}")
    return model.to(device).eval(), model_cfg


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8, top_k: int = 50, device: str = "cpu"):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    # v1.5.0: the terminator is the tokenizer's own end-of-document token.
    # The previous check looked for a token literally named "[EOS]", which no
    # tokenizer in this repository registers, so early stopping never fired.
    eos_id = tokenizer.token_to_id("<eos>")

    for _ in range(max_new_tokens):
        idx_cond = x if x.size(1) <= model.cfg["max_seq_len"] else x[:, -model.cfg["max_seq_len"] :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

        if eos_id is not None and next_id.item() == eos_id:
            break

    return tokenizer.decode(x[0].tolist())


def _apply_penalties(logits, generated, presence_penalty, frequency_penalty):
    """OpenAI-style presence/frequency penalties on already-seen tokens."""
    if not generated or (presence_penalty == 0.0 and frequency_penalty == 0.0):
        return logits
    counts = {}
    for t in generated:
        counts[t] = counts.get(t, 0) + 1
    idx = torch.tensor(list(counts.keys()), dtype=torch.long, device=logits.device)
    penalty = torch.tensor(
        [presence_penalty + frequency_penalty * counts[int(t)] for t in idx.tolist()],
        dtype=logits.dtype,
        device=logits.device,
    )
    logits = logits.clone()
    logits[0, idx] -= penalty
    return logits


def _filter_logits(logits, temperature, top_k, top_p):
    """Temperature, top-k and nucleus (top-p) filtering for one position."""
    logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens whose *preceding* cumulative mass already exceeds top_p,
        # always keeping the single most likely token.
        remove = (cumulative - sorted_probs) > top_p
        # v1.8.0: the nucleus mask belongs to the vocabulary dimension, and the
        # most likely token must always survive. Scattering along dimension 0
        # raised "index out of bounds" for any request with top_p < 1.
        remove[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
    return probs


@torch.no_grad()
def generate_ids(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    stop: list | None = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    device: str = "cpu",
    seed: int | None = None,
    binding_repair: bool = True,
    use_kv_cache: bool = False,
):
    """Single shared sampling implementation for every serving surface.

    Returns a dict with ``text`` (completion only, prompt stripped), ``token_ids``,
    ``finish_reason`` (``"stop"`` for EOS or a stop sequence, ``"length"`` when the
    token budget is exhausted), prompt/completion token counts and a structured
    ``repair`` diagnostic.

    The repair is deterministic, configurable and fail-closed. Ambiguous,
    unsafe, malformed or whole-function outputs are left as the raw completion.
    ``binding_repair=False`` is retained for A/B qualification. Generated code
    is never executed by this layer.

    When ``use_kv_cache=True`` (default), the first forward pass processes the
    full prompt and caches key/value tensors per layer; subsequent steps run only
    the new token against the cache. This is mathematically identical to the
    no-cache path but faster at longer context.
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt).ids
    prompt_tokens = len(prompt_ids)
    max_seq_len = model.cfg["max_seq_len"]
    eos_id = tokenizer.token_to_id("<eos>")
    stop_list = [stop] if isinstance(stop, str) else list(stop or [])
    generated = []
    finish_reason = "length"

    if seed is not None:
        torch.manual_seed(seed)

    if use_kv_cache:
        # ---- KV-cache path -------------------------------------------------
        kv_cache = KVCache(n_layers=model.cfg["n_layers"])

        # Step 0: process the full prompt through the model to populate cache.
        x_full = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        if x_full.size(1) > max_seq_len:
            x_full = x_full[:, -max_seq_len:]
        logits, _ = model(
            x_full,
            kv_cache=kv_cache,
            use_cache=True,
            position_offset=0,
        )
        next_logits = logits[:, -1, :]

        # Apply penalties and sample the first token.
        next_logits = _apply_penalties(
            next_logits, generated, presence_penalty, frequency_penalty
        )
        if temperature and temperature > 0:
            probs = _filter_logits(next_logits, temperature, top_k, top_p)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        tok = int(next_id.item())

        if tok == eos_id:
            finish_reason = "stop"
        else:
            generated.append(tok)
            x_new = next_id  # single new token for subsequent steps
            pos = prompt_tokens  # absolute position of the next token

            for _ in range(1, max_new_tokens):
                if pos >= max_seq_len:
                    # Window is full; stop generating to avoid RoPE misalignment.
                    break

                logits, _ = model(
                    x_new,
                    kv_cache=kv_cache,
                    use_cache=True,
                    position_offset=pos,
                )
                next_logits = logits[:, -1, :]
                next_logits = _apply_penalties(
                    next_logits, generated, presence_penalty, frequency_penalty
                )
                if temperature and temperature > 0:
                    probs = _filter_logits(next_logits, temperature, top_k, top_p)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
                tok = int(next_id.item())

                if tok == eos_id:
                    finish_reason = "stop"
                    break

                generated.append(tok)
                x_new = next_id
                pos += 1

                if stop_list:
                    tail = tokenizer.decode(generated)
                    hit = next((s for s in stop_list if s and s in tail), None)
                    if hit is not None:
                        generated = tokenizer.encode(tail[: tail.index(hit)]).ids
                        finish_reason = "stop"
                        break

        completion = tokenizer.decode(generated)

    else:
        # ---- No-cache path (original behaviour) ----------------------------
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            idx_cond = x if x.size(1) <= max_seq_len else x[:, -max_seq_len:]
            logits, _ = model(idx_cond)
            next_logits = _apply_penalties(
                logits[:, -1, :], generated, presence_penalty, frequency_penalty
            )
            if temperature and temperature > 0:
                probs = _filter_logits(next_logits, temperature, top_k, top_p)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            tok = int(next_id.item())

            if tok == eos_id:
                finish_reason = "stop"
                break

            generated.append(tok)
            x = torch.cat([x, next_id], dim=1)

            if stop_list:
                tail = tokenizer.decode(generated)
                hit = next((s for s in stop_list if s and s in tail), None)
                if hit is not None:
                    generated = tokenizer.encode(tail[: tail.index(hit)]).ids
                    finish_reason = "stop"
                    break

        completion = tokenizer.decode(generated)

    raw_completion = completion
    repair_diagnostic: dict = {
        "enabled": bool(binding_repair),
        "applied": False,
        "status": "repaired" if binding_repair else "disabled",
        "reason": "repair enabled" if binding_repair else "repair disabled",
        "mapping": {},
        "changed": False,
        "raw_preserved": True,
    }
    if binding_repair:
        try:
            from training.binding_repair import repair_from_prompt_info

            info = repair_from_prompt_info(prompt, completion)
            completion = info.completion
            repair_diagnostic.update({
                "applied": bool(info.changed),
                "status": info.status,
                "reason": info.reason,
                "mapping": dict(info.mapping),
                "changed": bool(info.changed),
            })
        except Exception as exc:  # keep raw output and explain the failure
            repair_diagnostic.update({
                "applied": False,
                "status": "repair_failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "changed": False,
            })
            completion = raw_completion
    return {
        "text": completion,
        "raw_text": raw_completion,
        "token_ids": generated,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(generated),
        "eos_id": eos_id,
        "repair": repair_diagnostic,
    }


def main():
    p = argparse.ArgumentParser(description="Taksh local inference")
    p.add_argument("--ckpt", required=True, help="path to checkpoint .pt file")
    p.add_argument("--tokenizer", default="tokenizer/taksh_code_bpe_v1.json")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    model, model_cfg = load_model(args.ckpt, device)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)

    def respond(message: str, history):
        if not message.strip():
            yield ""
            return
        prompt = f"Q: {message}\nA:"
        full = ""
        for chunk in generate(model, tokenizer, prompt, max_new_tokens=200, device=device):
            full = chunk
            yield chunk
        yield full + "\n\n---\n*Educational only. Not investment advice.*"

    import gradio as gr

    with gr.Blocks() as ui:
        gr.ChatInterface(
            fn=respond,
            title="Taksh — Your Own LLM",
            description=f"Trained from scratch on your corpus. {model_cfg['d_model']}d × {model_cfg['n_layers']}L = ~{model.num_params():,} params. Device: {device}.",
            examples=[
                "What is a P/E ratio?",
                "Explain CAGR with an example.",
                "What is compound interest?",
            ],
        )

    print(f"Starting Taksh UI on http://localhost:{args.port}")
    ui.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
