"""Zero-shot image-to-caption evaluation.

Measures M1 gate criterion "zero-shot caption works": for each image, run the
trained model and record the greedy caption + per-token logp of the generated
sequence.  This is measurement only -- it reports what the model produced
without judging quality and without any promotion decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import math

import torch
import torch.nn.functional as F

from evaluation.generate import (
    _prefix_embeds,
    decode_ids,
    generate_caption,
    image_tensor_file,
)
from training.multimodal import build_mm_mask
from training.mm_manifest import load_manifest


def evaluate_image(model, tokenizer, image, max_new_tokens: int = 32) -> dict:
    """Generate a caption for one preprocessed ``(1,3,H,W)`` image tensor.

    Returns a record with the decoded caption, the raw token ids and the
    mean per-token log-probability of the *generated* (non-EOS) tokens.
    """
    ids = generate_caption(model, image, max_new_tokens=max_new_tokens)

    if ids:
        embeds = _prefix_embeds(model, image, None)
        tok_ids = torch.tensor([ids], dtype=torch.long, device=image.device)
        with torch.no_grad():
            embeds = torch.cat([embeds, model.text.embed(tok_ids)], dim=1)
        mask = build_mm_mask([model.image_region], embeds.shape[1]).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).to(image.device)
        with torch.no_grad():
            logits, _ = model.text.decode(embeds, targets=None, attn_mask=mask)
        pos0 = model.cap_start_pos
        shifted = logits[0, pos0:pos0 + len(ids)].float()
        target = torch.tensor(ids, dtype=torch.long, device=image.device)
        per_token = -F.cross_entropy(shifted.view(-1, shifted.size(-1)), target.view(-1),
                                     reduction="none")
        mean_logp = float(per_token.mean().item())
    else:
        mean_logp = None

    return {
        "caption": decode_ids(tokenizer, ids),
        "tokens": ids,
        "generated_len": len(ids),
        "mean_logp": round(mean_logp, 4) if mean_logp is not None else None,
        "perplexity": round(math.exp(-mean_logp), 4) if mean_logp is not None else None,
    }


def evaluate_manifest(model, tokenizer, manifest_path, split="train", max_samples=None,
                      image_size=112, max_new_tokens=32, device="cpu"):
    """Evaluate captions for an image manifest (data/mm_m1/MANIFEST.json).

    Returns the list of per-sample records along with a short summary.  Missing
    image files are reported per-sample as errors and skipped.
    """
    manifest = load_manifest(manifest_path)
    samples = manifest.for_split(split)
    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]
    records = []
    errors = []
    for rec in samples:
        img_path = rec.resolved_path(Path(manifest_path).parent)
        if not Path(img_path).is_file():
            errors.append({"sample_id": rec.sample_id, "error": f"image file not found: {img_path}"})
            continue
        try:
            image = image_tensor_file(img_path, size=image_size, device=device)
            rep = evaluate_image(model, tokenizer, image, max_new_tokens=max_new_tokens)
            rep["sample_id"] = rec.sample_id
            records.append(rep)
        except Exception as exc:  # noqa: BLE001 - report per-sample and continue
            errors.append({"sample_id": rec.sample_id, "error": str(exc)})
    summary = {
        "split": split,
        "requested": len(samples),
        "evaluated": len(records),
        "errors": len(errors),
        "generated_any_tokens": sum(1 for r in records if r["tokens"]),
        "measurement_only": True,
    }
    return {"summary": summary, "records": records, "errors": errors}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.eval_caption",
        description="Zero-shot image-to-caption evaluation (measurement only).",
    )
    parser.add_argument("--ckpt", required=True, help="MM checkpoint (.pt)")
    parser.add_argument("--tokenizer", default="tokenizer/qm_bpe_v0.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--out", help="write JSON report here")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="evaluate a single image file")
    src.add_argument("--manifest", help="evaluate all images in a MANIFEST.json")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)

    from evaluation.mm_checkpoint import load_checkpoint
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    loaded = load_checkpoint(args.ckpt, device=args.device)
    if not loaded.is_mm:
        raise SystemExit(f"error: {args.ckpt} is not a multimodal checkpoint")

    if args.image is not None:
        image = image_tensor_file(args.image, size=args.image_size, device=args.device)
        rep = evaluate_image(loaded.model, tokenizer, image, max_new_tokens=args.max_new_tokens)
        report = {"type": "single_image", "image": args.image, "record": rep,
                  "note": "measurement only; no quality claim", "checkpoint": str(loaded.path)}
    else:
        report = evaluate_manifest(
            loaded.model, tokenizer, args.manifest, split=args.split,
            max_samples=args.max_samples, image_size=args.image_size,
            max_new_tokens=args.max_new_tokens, device=args.device,
        )
        report["checkpoint"] = str(loaded.path)
        report["note"] = "measurement only; no quality claim"

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    if args.image is None:
        s = report["summary"]
        print(f"requested={s['requested']} evaluated={s['evaluated']} errors={s['errors']} "
              f"with_tokens={s['generated_any_tokens']} (measurement only)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())