import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.load_eval import load_eval_set


def format_item(item: dict) -> str:
    return f"Q: {item['question']}\nA: {item['reference']}"


def load_qm_checkpoint(path: str, device: str):
    from training.model import QMModel

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("cfg") or ckpt.get("config", {})
    if "model" in raw_cfg and isinstance(raw_cfg["model"], dict):
        model_cfg = raw_cfg["model"]
    elif "vocab_size" in raw_cfg:
        model_cfg = raw_cfg
    else:
        import yaml
        model_path = raw_cfg.get("run", {}).get("model_config", "configs/model/code_30m.yaml")
        with open(model_path) as f:
            model_cfg = yaml.safe_load(f)["model"]
    model_cfg = dict(model_cfg)
    model_cfg.setdefault("rope_theta", 10000.0)
    model_cfg.setdefault("norm_eps", 1e-5)
    model = QMModel(model_cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), model_cfg


@torch.no_grad()
def heldout_ppl(
    model,
    tokens_path: str,
    seq_len: int = 2048,
    max_windows: int = 32,
    val_fraction: float = 0.01,
    device: str = "cpu",
):
    """Held-out NLL and perplexity; returns ``(None, None)`` when unmeasurable.

    Same false-pass guard as `evaluation.eval_code.compute_code_perplexity`: a
    held-out tail shorter than one window used to yield ``exp(0) == 1.0``.
    """
    data = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    split = int(len(data) * (1.0 - val_fraction))
    val_tokens = len(data) - split
    if val_tokens < seq_len + 1:
        return None, None

    losses = []
    for i in range(max_windows):
        off = split + i * seq_len
        if off + seq_len + 1 > len(data):
            break
        x = torch.from_numpy(data[off : off + seq_len].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(data[off + 1 : off + 1 + seq_len].astype(np.int64)).unsqueeze(0).to(device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())

    if not losses:
        return None, None
    avg = sum(losses) / len(losses)
    return avg, math_exp(avg)


def math_exp(x: float) -> float:
    import math

    return math.exp(min(20.0, x))


@torch.no_grad()
def score_seed(model, tokenizer_encode, device: str) -> list[dict]:
    results = []
    for item in load_eval_set():
        ids = torch.tensor([tokenizer_encode(format_item(item))], dtype=torch.long, device=device)
        _, loss = model(ids, targets=ids)
        results.append({"id": item["id"], "category": item["category"], "nll_per_token": round(loss.item(), 4)})
    return results


@torch.no_grad()
def score_seed_hf(model_name: str, device: str) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    results = []
    for item in load_eval_set():
        ids = tok(format_item(item), return_tensors="pt").input_ids.to(device)
        out = model(ids, labels=ids)
        results.append({"id": item["id"], "category": item["category"], "nll_per_token": round(out.loss.item(), 4)})
    return results


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-item NLL into a category summary.

    An empty ``rows`` list returns ``overall_mean_nll: None`` and ``ok: False``
    instead of ``0.0``. Dividing by ``max(1, len(rows))`` used to report a *perfect*
    mean NLL of 0.0 for an unevaluated seed set, which would satisfy the
    ``no_regression`` gate in ``evaluation/promotion.yaml``. Same false-pass class
    as the empty held-out-token split guarded in ``heldout_ppl``.
    """
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["nll_per_token"])
    if not rows:
        return {
            "overall_mean_nll": None,
            "by_category": {},
            "n_items": 0,
            "ok": False,
            "error": "no seed items were evaluated",
        }
    return {
        "overall_mean_nll": round(sum(r["nll_per_token"] for r in rows) / len(rows), 4),
        "by_category": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_cat.items())},
        "n_items": len(rows),
        "ok": True,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate a pilot checkpoint or HF benchmark rival")
    p.add_argument("--ckpt", default=None, help="path to qm_pilot_final.pt")
    p.add_argument("--hf-model", default=None, help="benchmark rival, e.g. Qwen/Qwen2.5-0.5B")
    p.add_argument("--tokens", default="data/corpus_v0.tokens.bin")
    p.add_argument("--out", default=None)
    p.add_argument("--skip-seed", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {}

    if args.ckpt:
        model, cfg = load_qm_checkpoint(args.ckpt, device)
        ppl_loss, ppl = heldout_ppl(model, args.tokens, device=device)
        if ppl_loss is None:
            report["heldout"] = {
                "nll": None,
                "perplexity": None,
                "ok": False,
                "error": "held-out split too small to score (need >= seq_len+1 tokens)",
            }
            print("held-out perplexity NOT measurable: split too small to score")
        else:
            report["heldout"] = {"nll": round(ppl_loss, 4), "perplexity": round(ppl, 2), "ok": True}
            print(f"held-out NLL {ppl_loss:.4f} | ppl {ppl:.2f}")
        if not args.skip_seed:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file("tokenizer/qm_bpe_v0.json")
            rows = score_seed(model, lambda s: tok.encode(s).ids, device)
            report["india_finance_seed"] = {"rows": rows, "summary": summarize(rows)}
            print(json.dumps(report["india_finance_seed"]["summary"], indent=2))
    elif args.hf_model:
        rows = score_seed_hf(args.hf_model, device)
        report["hf_model"] = args.hf_model
        report["india_finance_seed"] = {"rows": rows, "summary": summarize(rows)}
        print(args.hf_model, json.dumps(report["india_finance_seed"]["summary"], indent=2))
    else:
        raise SystemExit("pass --ckpt or --hf-model")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
