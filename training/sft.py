import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from training.model import QMModel
from training.pretrain import load_checkpoint, lr_at, save_checkpoint

SYSTEM_PROMPT = (
    "You are Quantsmind Coder, an expert programming assistant specializing in "
    "Python and JavaScript/TypeScript. You help with code generation, debugging, "
    "code review, refactoring, and technical explanations.\n"
    "- Always produce correct, idiomatic code with proper error handling.\n"
    "- Explain your reasoning before providing code solutions.\n"
    "- When reviewing code, identify bugs, security issues, and performance problems.\n"
    "- Follow PEP 8 for Python and standard conventions for JavaScript/TypeScript.\n"
    "- If you are unsure about an API or library version, state your assumptions."
)

TEMPLATE = "<|system|>{system}\n<|user|>{instruction}\n<|assistant|>{response}<|eos|>"


class SFTPairDataset:
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pairs.append(json.loads(line))
        if not self.pairs:
            raise ValueError(f"No SFT pairs found in {jsonl_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        instruction = pair.get("instruction", pair.get("input", ""))
        response = pair.get("response", pair.get("output", ""))
        text = TEMPLATE.format(
            system=SYSTEM_PROMPT,
            instruction=instruction,
            response=response,
        )
        ids = self.tokenizer.encode(text).ids
        ids = ids[: self.max_length]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        prompt_text = TEMPLATE.format(
            system=SYSTEM_PROMPT, instruction=instruction, response=""
        )
        prompt_ids = len(self.tokenizer.encode(prompt_text).ids) - 1
        labels[:prompt_ids] = -100
        return input_ids, labels


def collate_sft(batch, pad_id=0):
    max_len = max(len(x[0]) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        input_ids[i, : len(x)] = x
        labels[i, : len(y)] = y
    return input_ids, labels


def main():
    p = argparse.ArgumentParser(description="Quantsmind SFT training loop")
    p.add_argument("--config", required=True)
    p.add_argument("--sft-data", required=True)
    p.add_argument("--out-dir", default="runs/sft")
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    )
    print(f"device={device}")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    dataset = SFTPairDataset(args.sft_data, tokenizer, cfg["data"]["seq_len"])
    print(f"SFT pairs: {len(dataset)}")

    model_cfg = load_yaml(cfg["run"]["model_config"])["model"]
    model = QMModel(model_cfg).to(device)

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"initialized from {args.init_ckpt}")

    opt_cfg = cfg["optimizer"]
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr_peak"],
        betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.95)),
        weight_decay=opt_cfg.get("weight_decay", 0.1),
    )

    train_cfg = cfg["training"]
    total_steps = args.max_steps or train_cfg["max_steps"]
    start_step = 0
    last_ckpt = out_dir / "checkpoint_last.pt"
    if args.resume and last_ckpt.exists():
        start_step = load_checkpoint(last_ckpt, model, optim)
        print(f"resumed from {last_ckpt} at step {start_step}")

    if device == "cuda":
        torch.compile(model)

    batch_cfg = cfg["batch"]
    micro_bs = batch_cfg["micro_batch_size"]
    accum = batch_cfg.get("grad_accum_steps", 1)
    seq_len = cfg["data"]["seq_len"]

    log_every = train_cfg.get("log_every_steps", 10)
    ckpt_every = train_cfg.get("checkpoint_every_steps", 500)
    ema_loss = None
    model.train()
    t0 = time.time()

    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, opt_cfg["lr_peak"], opt_cfg.get("warmup_steps", 0), opt_cfg.get("min_lr_ratio", 0.1))
        for g in optim.param_groups:
            g["lr"] = lr

        loss_val = 0.0
        optim.zero_grad(set_to_none=True)
        for _ in range(accum):
            batch = [dataset[i % len(dataset)] for i in range(micro_bs)]
            x, y = collate_sft(batch)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            (loss / accum).backward()
            loss_val += loss.item() / accum

        grad_clip = opt_cfg.get("grad_clip")
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        if not math.isfinite(loss_val):
            raise RuntimeError(f"non-finite loss {loss_val} at step {step}")

        ema = 0.05 if ema_loss is None else 0.95
        ema_loss = ema * loss_val + (1 - ema) * (ema_loss or loss_val)

        if step % log_every == 0:
            print(f"step {step}/{total_steps} loss={loss_val:.4f} ema={ema_loss:.4f} lr={lr:.2e}")

        if (step + 1) % ckpt_every == 0 or (step + 1) == total_steps:
            save_checkpoint(last_ckpt, model, optim, step + 1, cfg)
            print(f"checkpoint saved: {last_ckpt} (step {step + 1})")

    print("SFT training complete")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
