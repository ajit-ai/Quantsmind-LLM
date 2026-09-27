import argparse
import math
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
import yaml

from training.image_data import ImageTextDataset, RealImageDataset
from training.multimodal import QMMultimodal, load_text_state, probe_conditioning
from training.pretrain import load_yaml, lr_at


def grad_norm(module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.float().pow(2).sum().item()
    return math.sqrt(total)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description="Quantsmind multimodal fit-check (M1.1 synthetic / M1.2 real-data)")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", default="runs/mm_fit_check")
    p.add_argument("--data", default=None, help=".npz from scripts/synth_images.py")
    p.add_argument("--manifest", default=None, help="MANIFEST.json for the M1.2 real-image pipeline")
    p.add_argument("--split", default="train", help="manifest split to train on (default: train)")
    p.add_argument("--image-size", type=int, default=None, help="override image resolution")
    p.add_argument("--init-ckpt", default=None, help="M0 text checkpoint to warm-start from")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    )
    seed = args.seed if args.seed is not None else cfg.get("run", {}).get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)  # deterministic CPU fit-check

    run_cfg = cfg["run"]
    text_cfg = load_yaml(run_cfg["model_config"])["model"]
    vision_cfg = load_yaml(run_cfg["vision_config"])["vision"]
    assert vision_cfg["d_model"] == text_cfg["d_model"], "text/vision d_model must match"

    data_path = args.data or cfg.get("data", {}).get("synthetic_path", "runs/mm_synth/data.npz")
    tokenizer_path = cfg.get("data", {}).get("tokenizer", "tokenizer/qm_bpe_v0.json")
    manifest_path = args.manifest or cfg.get("data", {}).get("train_manifest")
    preprocessing_version = cfg.get("image", {}).get("preprocessing_version", "mm1_v1")
    image_size = args.image_size or vision_cfg.get("image_size", 112)
    if args.manifest or manifest_path:
        split = (args.split or cfg.get("data", {}).get("train_split") or "train")
        ds = RealImageDataset(
            manifest_path,
            split=split,
            tokenizer_path=tokenizer_path,
            image_size=image_size,
            normalize=cfg.get("image", {}).get("normalize"),
            seed=seed,
        )
        data_kind = f"manifest:{Path(manifest_path).name}:{split}"
    else:
        if image_size != vision_cfg.get("image_size", 112):
            raise ValueError("--image-size requires --manifest (synthetic data is fixed at config resolution)")
        ds = ImageTextDataset(data_path, tokenizer_path=tokenizer_path, seed=seed)
        data_kind = f"synthetic:{Path(data_path).name}"

    model = QMMultimodal(text_cfg, vision_cfg).to(device)
    init_ckpt = args.init_ckpt
    if init_ckpt is None:
        init_ckpt = cfg.get("run", {}).get("init_ckpt")
    if init_ckpt and Path(init_ckpt).exists():
        new_params = load_text_state(model, init_ckpt)
        print(f"text warm-start: {init_ckpt} (new/ignored params: {len(new_params)})")
    else:
        print("text init: random (no M0 checkpoint available)")

    n_text = model.text.num_params(non_embedding=False)
    n_vision = model.vision.num_params()
    n_proj = sum(x.numel() for x in model.projector.parameters())
    print(
        f"device={device} params=text {n_text:,} + vision {n_vision:,} + projector {n_proj:,} "
        f"= {model.num_params(non_embedding=False):,} total"
    )

    opt_cfg = cfg["optimizer"]
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr_peak"],
        betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.95)),
        weight_decay=opt_cfg.get("weight_decay", 0.1),
    )

    batch_cfg = cfg["batch"]
    micro_bs = batch_cfg["micro_batch_size"]
    accum = batch_cfg.get("grad_accum_steps", 1)
    total_steps = args.max_steps or cfg["training"]["max_steps"]
    log_every = cfg["training"].get("log_every_steps", 25)
    nan_every = cfg.get("watchdog", {}).get("nan_check_every_steps", 25)
    grad_clip = opt_cfg.get("grad_clip")
    seq_tokens = 3 + model.image_tokens + ds.max_caption_len + 1  # +1 caption slot for EOS target

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    t0 = time.time()
    ema_loss = None
    vit_norms, proj_norms, losses, lrs = [], [], [], []
    for step in range(total_steps):
        lr = lr_at(step, total_steps, opt_cfg["lr_peak"], opt_cfg.get("warmup_steps", 0), opt_cfg.get("min_lr_ratio", 0.1))
        for g in optim.param_groups:
            g["lr"] = lr

        loss_val = 0.0
        optim.zero_grad(set_to_none=True)
        for _ in range(accum):
            images, caption_ids, lengths = ds.batch(micro_bs)
            images, caption_ids, lengths = images.to(device), caption_ids.to(device), lengths.to(device)
            _, loss = model(images, caption_ids, lengths)
            (loss / accum).backward()
            loss_val += loss.item() / accum

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if not math.isfinite(loss_val):
            raise RuntimeError(f"non-finite loss {loss_val} at step {step}; aborting per watchdog policy")

        optim.step()
        losses.append(loss_val)
        lrs.append(lr)
        if step % nan_every == 0:
            vit_norms.append(grad_norm(model.vision))
            proj_norms.append(grad_norm(model.projector))

        ema = 0.05 if ema_loss is None else 0.95
        ema_loss = ema * loss_val + (1 - ema) * (ema_loss or loss_val)

        if step % log_every == 0:
            steps_done = step + 1
            rate = micro_bs * accum * seq_tokens * steps_done / (time.time() - t0)
            print(
                f"step {step:4d}/{total_steps} loss={loss_val:.4f} ema={ema_loss:.4f} lr={lr:.2e} "
                f"grad(viT)={grad_norm(model.vision):.4f} grad(proj)={grad_norm(model.projector):.4f} "
                f"grad(total)={grad_norm(model):.4f} tok/s={rate:.0f} elapsed={(time.time()-t0)/60:.1f}min"
            )

    elapsed = time.time() - t0
    final_loss = losses[-1] if losses else float("nan")
    init_loss = losses[0] if losses else float("nan")

    # Conditioning probe: two different images, same caption prefix.
    imgs = []
    cap_ids = None
    for i in (0, 1):
        images, caption_ids, lengths = ds.batch(1)
        imgs.append(images[0:1].to(device))
        if cap_ids is None:
            cap_ids = caption_ids
    diff, arg_a, arg_b = probe_conditioning(model, imgs[0], imgs[1], cap_ids.to(device))
    print(f"conditioning probe: max|logit diff|={diff:.6f} argmax_a={arg_a} argmax_b={arg_b}")

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "step": total_steps,
        # Compatibility: pretrain.save_checkpoint stores the training config under
        # "config" (local_inference reads ckpt["config"]["run"]["model_config"]) and
        # evaluate_pilot reads ckpt["cfg"] as a flat model config. Preserve both.
        "config": cfg,
        "cfg": text_cfg,
    }
    meta = {
        "kind": data_kind,
        "seed": seed,
        "preprocessing_version": preprocessing_version,
        "image_size": image_size,
        "max_steps": total_steps,
        "torch_version": torch.__version__,
        "code_commit": git_commit(),
        "manifest": None,
    }
    if isinstance(ds, RealImageDataset):
        meta["manifest"] = {
            "path": str(ds.manifest.path),
            "sha256": ds.manifest.sha256(),
            "samples": len(ds.samples),
            "split": ds.split,
        }
        print(
            f"manifest identity: sha256={meta['manifest']['sha256'][:12]} samples={len(ds.samples)} "
            f"split={ds.split}"
        )
    ckpt["metadata"] = meta  # additive: existing keys are untouched
    ckpt_path = out_dir / "checkpoint_last.pt"
    torch.save(ckpt, ckpt_path)
    print(f"saved {ckpt_path}")

    print(
        f"FIT-CHECK init_loss={init_loss:.4f} final_loss={final_loss:.4f} "
        f"eps=ln(49156)~{math.log(49156):.2f} target<8.0 elapsed={elapsed:.1f}s "
        f"vit_grad_final={vit_norms[-1] if vit_norms else 0.0:.4f} "
        f"proj_grad_final={proj_norms[-1] if proj_norms else 0.0:.4f}"
    )


if __name__ == "__main__":
    main()