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


def load_model(checkpoint_path: str, device: str):
    from training.model import QMModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]["run"]["model_config"] if "config" in ckpt else None
    if cfg:
        import yaml

        with open(cfg) as f:
            model_cfg = yaml.safe_load(f)["model"]
    else:
        model_cfg = ckpt.get("model_cfg", ckpt.get("config", {}))
        if "model" in model_cfg:
            model_cfg = model_cfg["model"]

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


def main():
    p = argparse.ArgumentParser(description="Quantsmind local inference")
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
            title="Quantsmind — Your Own LLM",
            description=f"Trained from scratch on your corpus. {model_cfg['d_model']}d × {model_cfg['n_layers']}L = ~{model.num_params():,} params. Device: {device}.",
            examples=[
                "What is a P/E ratio?",
                "Explain CAGR with an example.",
                "What is compound interest?",
            ],
        )

    print(f"Starting Quantsmind UI on http://localhost:{args.port}")
    ui.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
