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
    cfg = ckpt["cfg"]
    model_cfg = {
        "vocab_size": cfg["vocab_size"],
        "d_model": cfg["d_model"],
        "n_layers": cfg["n_layers"],
        "n_heads": cfg["n_heads"],
        "n_kv_heads": cfg["n_kv_heads"],
        "ffn_dim": cfg["ffn_dim"],
        "rope_theta": cfg.get("rope_theta", 10000.0),
        "norm_eps": cfg.get("norm_eps", 1e-5),
    }
    model = QMModel(model_cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def heldout_ppl(model, tokens_path: str, seq_len: int = 2048, max_windows: int = 32, device: str = "cpu"):
    data = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    split = int(len(data) * 0.99)
    losses = []
    for i in range(max_windows):
        off = split + i * seq_len
        if off + seq_len + 1 > len(data):
            break
        x = torch.from_numpy(data[off : off + seq_len].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(data[off + 1 : off + 1 + seq_len].astype(np.int64)).unsqueeze(0).to(device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    avg = sum(losses) / max(1, len(losses))
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
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["nll_per_token"])
    return {
        "overall_mean_nll": round(sum(r["nll_per_token"] for r in rows) / max(1, len(rows)), 4),
        "by_category": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_cat.items())},
        "n_items": len(rows),
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
        report["heldout"] = {"nll": round(ppl_loss, 4), "perplexity": round(ppl, 2)}
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
