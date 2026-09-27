"""Text-task evaluation for the multimodal models (regression/forgetting check).

Reuses the pilot metrics from ``evaluation/evaluate_pilot.py`` (held-out
perplexity, seed-question set NLL by category) but is MM-aware: it extracts
the text component from a multimodal checkpoint (or uses a text-only
checkpoint directly), so the same metrics can be run on both an MM checkpoint
and the text-only reference it was warm-started from.  Reported deltas are
measurements, not decisions -- thresholds live in ``promotion.yaml``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.evaluate_pilot import heldout_ppl, summarize
from evaluation.load_eval import load_eval_set
from evaluation.mm_checkpoint import load_checkpoint


@torch.no_grad()
def score_items(model, items, tok_encode, device: str = "cpu") -> list[dict]:
    """Score seed items exactly like ``evaluate_pilot.score_seed``."""
    rows = []
    for item in items:
        ids = torch.tensor(
            [tok_encode(f"Q: {item['question']}\nA: {item['reference']}")],
            dtype=torch.long, device=device,
        )
        _, loss = model(ids, targets=ids)
        rows.append({
            "id": item["id"],
            "category": item["category"],
            "nll_per_token": round(float(loss.item()), 4),
        })
    return rows


def evaluate_text(
    model,
    tok_encode,
    seed_path: str | None,
    tokens_path: str | None,
    device: str = "cpu",
    seq_len: int = 2048,
    max_windows: int = 32,
) -> dict:
    """Run the text metrics for one text model.

    Returns a dict of metric sections.  Unavailable inputs produce ``None``
    markers rather than fabricated numbers.
    """
    rep = {}
    if tokens_path is not None and Path(tokens_path).is_file():
        nll, ppl = heldout_ppl(model, tokens_path, seq_len=seq_len,
                               max_windows=max_windows, device=device)
        rep["heldout_perplexity"] = ppl
        rep["heldout_tokens/ppl_windows"] = {"max_windows": max_windows, "nll": nll}
    else:
        rep["heldout_perplexity"] = None

    if seed_path is not None and Path(seed_path).is_file():
        items = load_eval_set(Path(seed_path))
        rows = score_items(model, items, tok_encode, device=device)
        rep["seed_summary"] = summarize(rows)
        rep["seed_items"] = rows
    else:
        rep["seed_summary"] = None

    return rep


def _deltas(a: dict, b: dict) -> dict:
    d = {}
    for key in ("heldout_perplexity",):
        if isinstance(a.get(key), (int, float)) and isinstance(b.get(key), (int, float)):
            d[key] = {"candidate": b[key], "reference": a[key], "delta": b[key] - a[key]}
    sa, sb = a.get("seed_summary"), b.get("seed_summary")
    if isinstance(sa, dict) and isinstance(sb, dict):
        d["seed_overall_nll_delta"] = {
            "candidate": sb.get("overall_mean_nll"),
            "reference": sa.get("overall_mean_nll"),
            "delta": (
                sb.get("overall_mean_nll") - sa.get("overall_mean_nll")
                if sa.get("overall_mean_nll") is not None
                and sb.get("overall_mean_nll") is not None else None
            ),
        }
    return d


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.eval_text",
        description="MM-aware text regression/forgetting evaluation (measurement only).",
    )
    parser.add_argument("--ckpt", required=True, help="MM or text-only checkpoint (.pt)")
    parser.add_argument("--ref", help="reference checkpoint to diff against (e.g. text-only M0)")
    parser.add_argument("--tokenizer", default="tokenizer/qm_bpe_v0.json")
    parser.add_argument("--tokens", help="uint16 memmap of eval tokens for heldout PPL")
    parser.add_argument("--seed", default="evaluation/india_finance/seed_v0.jsonl",
                        help="seed question set (JSONL, load_eval format)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-windows", type=int, default=32)
    parser.add_argument("--out", help="write JSON report here")
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    encode = lambda s: tok.encode(s).ids  # noqa: E731

    loaded = load_checkpoint(args.ckpt, device=args.device)
    rep = evaluate_text(
        loaded.text_model, encode, seed_path=args.seed, tokens_path=args.tokens,
        device=args.device, seq_len=args.seq_len, max_windows=args.max_windows,
    )
    rep["checkpoint"] = str(loaded.path)
    rep["checkpoint_is_mm"] = loaded.is_mm
    rep["note"] = "measurement only; no regression decision is made here"

    if args.ref is not None:
        ref = load_checkpoint(args.ref, device=args.device)
        ref_rep = evaluate_text(
            ref.text_model, encode, seed_path=args.seed, tokens_path=args.tokens,
            device=args.device, seq_len=args.seq_len, max_windows=args.max_windows,
        )
        rep["reference"] = {
            "checkpoint": str(ref.path),
            "checkpoint_is_mm": ref.is_mm,
            "deltas_vs_reference": _deltas(ref_rep, rep),
        }

    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    hp = rep.get("heldout_perplexity")
    ss = rep.get("seed_summary") or {}
    print(f"heldout_ppl={hp} seed_overall_nll={ss.get('overall_mean_nll')} (measurement only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())