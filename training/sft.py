import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from training.chat_template import (
    CHAT_TEMPLATE,
    SYSTEM_PROMPT,
    IncompatibleTokenizer,
    prompt_token_length,
    render_completion,
    render_prompt,
    validate_chat_template,
)
from training.checkpoint import CheckpointIncompatible, init_from_checkpoint
from training.loss_log import LossLogger
from training.model import QMModel
from training.pretrain import load_checkpoint, lr_at, save_checkpoint

# Re-exported for existing callers/tests. TEMPLATE is now rendered from tokens the
# tokenizer actually registers (C3); the old "<|system|>..." form shredded into
# byte tokens because none of those markers existed in the vocabulary.
TEMPLATE = CHAT_TEMPLATE

QUARANTINE_SUFFIX = ".quarantine.json"


def find_quarantine(sft_data_path: str) -> Path | None:
    """Return the quarantine marker sitting beside an SFT dataset, if any.

    A dataset is quarantined when ``<stem>.quarantine.json`` exists next to it, e.g.
    ``data/code_sft_v0.jsonl`` -> ``data/code_sft_v0.quarantine.json``.
    """
    marker = Path(sft_data_path).with_suffix(QUARANTINE_SUFFIX)
    return marker if marker.exists() else None


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
        # C3: refuse a tokenizer that cannot represent the template as single
        # tokens, rather than training on shredded byte fragments.
        self.markers = validate_chat_template(tokenizer)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        instruction = pair.get("instruction", pair.get("input", ""))
        response = pair.get("response", pair.get("output", ""))
        text = render_completion(SYSTEM_PROMPT, instruction, response)
        ids = self.tokenizer.encode(text).ids
        truncated = len(ids) > self.max_length
        ids = ids[: self.max_length]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        # Mask everything up to and including the prompt: supervision starts at
        # the response, and the final <eos> is what the model must learn to emit.
        prompt_ids = prompt_token_length(self.tokenizer, SYSTEM_PROMPT, instruction)
        prompt_ids = min(prompt_ids, labels.numel())
        labels[:prompt_ids] = -100
        return input_ids, labels


class SFTBatcher:
    """Shuffled, epoch-cycling index stream for SFT micro-batches.

    The training loop previously built each batch as
    ``[dataset[i % len(dataset)] for i in range(micro_bs)]``, where ``i`` only
    ever ranged over ``0..micro_bs-1``. Every step therefore saw the same first
    ``micro_bs`` pairs and the rest of the dataset was never trained on. This
    yields a fresh shuffle each epoch and refills mid-epoch.
    """

    def __init__(self, n_items: int, batch_size: int, seed: int = 42):
        if n_items <= 0:
            raise ValueError("cannot build batches from an empty dataset")
        self.n_items = n_items
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self._order: list[int] = []
        self._pos = 0
        # Coverage accounting (A1): the defect this class exists to fix was
        # invisible, so consumption is tracked explicitly.
        self.consumed = 0
        self.seen: set[int] = set()
        self.epochs_completed = 0
        self.refills = 0

    def _refill(self) -> None:
        self._order = list(range(self.n_items))
        self.rng.shuffle(self._order)
        if self._pos >= self.n_items:
            self.epochs_completed += 1
        self._pos = 0
        self.refills += 1

    def next_indices(self) -> list[int]:
        if not self._order:
            self._refill()
        out: list[int] = []
        while len(out) < self.batch_size:
            if self._pos >= self.n_items:
                self._refill()
            out.append(self._order[self._pos])
            self._pos += 1
        self.consumed += len(out)
        self.seen.update(out)
        return out

    @property
    def unique_coverage(self) -> float:
        """Fraction of the dataset that has been seen at least once."""
        return len(self.seen) / self.n_items if self.n_items else 0.0

    @property
    def epoch_float(self) -> float:
        return self.consumed / self.n_items if self.n_items else 0.0

    def coverage_report(self) -> dict:
        return {
            "dataset_size": self.n_items,
            "consumed_examples": self.consumed,
            "unique_examples_seen": len(self.seen),
            "unique_coverage": round(self.unique_coverage, 6),
            "epochs_completed": self.epochs_completed,
            "epochs_consumed": round(self.epoch_float, 6),
            "shuffle_refills": self.refills,
        }


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
    p.add_argument(
        "--init-ckpt-partial",
        action="store_true",
        help="allow --init-ckpt to initialise only the parameters it provides; "
        "the rest stay random and are reported. Shape mismatches still abort.",
    )
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--ignore-sft-quarantine",
        action="store_true",
        help="train on a quarantined SFT dataset; diagnostics only, no capability claim",
    )
    args = p.parse_args()

    marker = find_quarantine(args.sft_data)
    if marker and not args.ignore_sft_quarantine:
        try:
            reason = json.loads(marker.read_text(encoding="utf-8")).get("reason", "see marker file")
        except (json.JSONDecodeError, OSError):
            reason = "see marker file"
        raise SystemExit(
            f"SFT dataset is quarantined: {args.sft_data}\n"
            f"  {reason}\n"
            f"  marker: {marker}\n"
            "Refusing to train. Pass --ignore-sft-quarantine to override for "
            "diagnostics only; a checkpoint trained on it supports no capability claim."
        )

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
        # B3: this used to load with strict=False and print "initialized from ..."
        # regardless of the outcome, which is how runs/code_sft silently became a
        # randomly-initialised 495,808-parameter model. Compatibility is now
        # checked, and partial loading must be asked for explicitly.
        report = init_from_checkpoint(
            model, args.init_ckpt, allow_partial=args.init_ckpt_partial
        )
        print(f"initialized from {args.init_ckpt}")
        print(report.describe())
        if report.partial and report.missing:
            print(
                f"WARNING: {len(report.missing)} parameter(s) were NOT "
                "initialised from the checkpoint and remain random."
            )

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

    # A1: the loop used to build every batch as
    # ``[dataset[i % len(dataset)] for i in range(micro_bs)]``, so each step saw
    # the same first ``micro_bs`` pairs and the rest of the corpus was never
    # trained on. An explicit, shuffled, epoch-cycling batcher replaces it.
    seed = cfg.get("run", {}).get("seed", 42)
    batcher = SFTBatcher(len(dataset), micro_bs, seed=seed)
    if args.resume and start_step > 0:
        # Deterministic resume: replay the index stream the interrupted run would
        # have produced, so consumed_examples in the checkpoint matches reality
        # instead of restarting coverage from zero.
        batches_per_step = accum
        for _ in range(start_step * batches_per_step):
            batcher.next_indices()

    examples_per_step = micro_bs * accum
    logger = LossLogger(
        out_dir / "loss_log.jsonl",
        run_kind="sft",
        model_config=cfg["run"].get("model_config"),
        dataset=args.sft_data,
        config=cfg,
    )
    logger.log_header(
        total_steps=total_steps,
        dataset_size=len(dataset),
        examples_per_step=examples_per_step,
        seq_len=seq_len,
        seed=seed,
        init_ckpt=args.init_ckpt,
    )
    print(f"loss log: {out_dir / 'loss_log.jsonl'}")

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
            indices = batcher.next_indices()
            batch = [dataset[i] for i in indices]
            x, y = collate_sft(batch)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            (loss / accum).backward()
            loss_val += loss.item() / accum

        grad_clip = opt_cfg.get("grad_clip")
        grad_norm = None
        if grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
        optim.step()

        if not math.isfinite(loss_val):
            raise RuntimeError(f"non-finite loss {loss_val} at step {step}")

        ema = 0.05 if ema_loss is None else 0.95
        ema_loss = ema * loss_val + (1 - ema) * (ema_loss or loss_val)

        if step % log_every == 0 or (step + 1) == total_steps:
            logger.log_step(
                step,
                loss_val,
                epoch=batcher.epoch_float,
                lr=lr,
                ema_loss=ema_loss,
                grad_norm=float(grad_norm) if grad_norm is not None else None,
                consumed_examples=batcher.consumed,
            )
            print(
                f"step {step}/{total_steps} loss={loss_val:.4f} "
                f"ema={ema_loss:.4f} lr={lr:.2e} "
                f"consumed={batcher.consumed} unique={len(batcher.seen)}/{len(dataset)}"
            )

        if (step + 1) % ckpt_every == 0 or (step + 1) == total_steps:
            save_checkpoint(
                last_ckpt,
                model,
                optim,
                step + 1,
                cfg,
                extra={
                    "consumed_examples": batcher.consumed,
                    "epochs_consumed": batcher.epoch_float,
                    "dataset_size": len(dataset),
                    "coverage": batcher.coverage_report(),
                },
            )
            print(f"checkpoint saved: {last_ckpt} (step {step + 1})")

    coverage = batcher.coverage_report()
    coverage["total_steps"] = total_steps
    coverage["optimizer_steps_completed"] = total_steps - start_step
    coverage["examples_per_step"] = examples_per_step
    coverage["eligible_examples"] = len(dataset)
    logger.close(status="complete", coverage=coverage)
    (out_dir / "sft_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    print(f"coverage: {json.dumps(coverage)}")
    if coverage["unique_coverage"] < 1.0:
        print(
            f"WARNING: only {coverage['unique_coverage']:.1%} of the SFT dataset "
            "was seen. This schedule does not cover the full dataset; report it "
            "as a subset run rather than full-dataset training."
        )
    print("SFT training complete")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
