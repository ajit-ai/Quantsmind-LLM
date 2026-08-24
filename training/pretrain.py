import argparse
import math
import os
import time
from pathlib import Path

import torch
import yaml

from training.data import TokenDataset
from training.model import QMModel


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_model_cfg(train_cfg: dict) -> dict:
    model_cfg = load_yaml(train_cfg["run"]["model_config"])["model"]
    override = train_cfg.get("model", {}).get("vocab_size_override")
    if override:
        model_cfg["vocab_size"] = int(override)
    return model_cfg


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(path: Path, model, optimizer, step: int, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


def main():
    p = argparse.ArgumentParser(description="Quantsmind-LLM pre-training loop")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default="runs/pretrain")
    p.add_argument("--tokens", default=None, help="override token file path")
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

    data_cfg = cfg["data"]
    tokens_path = args.tokens or os.path.join("data", "tokens.bin")
    dataset = TokenDataset(tokens_path, seed=cfg.get("run", {}).get("seed", 42))

    model = QMModel(flatten_model_cfg(cfg)).to(device)
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
    seq_len = data_cfg["seq_len"]

    use_wandb = os.environ.get("QM_WANDB") == "1"
    if use_wandb:
        import wandb

        wandb.init(project="quantsmind-llm", name=cfg.get("run", {}).get("name"), config=cfg)

    nan_every = cfg.get("watchdog", {}).get("nan_check_every_steps", 100)
    log_every = train_cfg.get("log_every_steps", 10)
    ckpt_every = train_cfg.get("checkpoint_every_steps", 1000)
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
            x, y = dataset.batch(micro_bs, seq_len)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            (loss / accum).backward()
            loss_val += loss.item() / accum

        grad_clip = opt_cfg.get("grad_clip")
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        if not math.isfinite(loss_val):
            raise RuntimeError(f"non-finite loss {loss_val} at step {step}; aborting per watchdog policy")

        ema = 0.05 if ema_loss is None else 0.95
        ema_loss = ema * loss_val + (1 - ema) * (ema_loss or loss_val)

        if use_wandb:
            wandb.log({"loss": loss_val, "ema_loss": ema_loss, "lr": lr, "step": step})

        if step % log_every == 0:
            rate = micro_bs * accum * seq_len * max(1, step - start_step + 1) / (time.time() - t0)
            print(f"step {step}/{total_steps} loss={loss_val:.4f} ema={ema_loss:.4f} lr={lr:.2e} tok/s={rate:.0f}")

        if (step + 1) % ckpt_every == 0 or (step + 1) == total_steps:
            save_checkpoint(last_ckpt, model, optim, step + 1, cfg)
            print(f"checkpoint saved: {last_ckpt} (step {step + 1})")

    print("training complete")


if __name__ == "__main__":
    main()
