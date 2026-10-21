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


def torch_load_checkpoint(path: str):
    """Load a checkpoint with safe deserialisation, falling back loudly.

    ``torch.load`` executes pickle opcodes, so a checkpoint is executable code by
    construction. ``weights_only=True`` refuses to unpickle arbitrary objects and is
    attempted first. If a checkpoint genuinely cannot be read that way we fall back,
    but we say so on stderr rather than silently weakening the guarantee -- callers
    are expected to supply a trusted path (startup config, never an HTTP request).
    """
    import sys as _sys

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - fall back, but never silently
        print(
            f"WARNING: refusing weights_only=True for {path} ({type(exc).__name__}); "
            "falling back to weights_only=False. This executes pickle data, so the "
            "path MUST come from trusted startup configuration, never from a request.",
            file=_sys.stderr,
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


def load_model(checkpoint_path: str, device: str):
    from training.model import QMModel

    ckpt = torch_load_checkpoint(checkpoint_path)
    model_cfg = resolve_model_config(ckpt)

    model = QMModel(model_cfg)
    model.load_state_dict(ckpt["model"])
    step = ckpt.get("step", "?")
    print(f"loaded checkpoint step={step} params={model.num_params():,}")
    return model.to(device).eval(), model_cfg


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8, top_k: int = 50, device: str = "cpu"):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

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

        if next_id.item() == tokenizer.token_to_id("[EOS]") if hasattr(tokenizer, "token_to_id") else False:
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
        remove[0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
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
):
    """Single shared sampling implementation for every serving surface.

    Returns a dict with ``text`` (completion only, prompt stripped), ``token_ids``,
    ``finish_reason`` (``"stop"`` for EOS or a stop sequence, ``"length"`` when the
    token budget is exhausted) and accurate ``prompt_tokens`` / ``completion_tokens``
    counts.

    There is no KV cache in this model, so each step re-runs a full forward pass over
    the (left-truncated) window. That is correct but slow at long context; see
    docs/OPEN_ISSUES.md.
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt).ids
    prompt_tokens = len(prompt_ids)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    if seed is not None:
        torch.manual_seed(seed)

    max_seq_len = model.cfg["max_seq_len"]
    eos_id = tokenizer.token_to_id("<eos>")
    stop_list = [stop] if isinstance(stop, str) else list(stop or [])
    generated = []
    finish_reason = "length"

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
                # Trim the stop sequence and anything after it from the output.
                generated = tokenizer.encode(tail[: tail.index(hit)]).ids
                finish_reason = "stop"
                break

    completion = tokenizer.decode(generated)
    return {
        "text": completion,
        "token_ids": generated,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(generated),
        "eos_id": eos_id,
    }


def main():
    p = argparse.ArgumentParser(description="Taksh local inference")
    p.add_argument("--ckpt", required=True, help="path to checkpoint .pt file")
    p.add_argument("--tokenizer", default="tokenizer/qm_bpe_v0.json")
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
