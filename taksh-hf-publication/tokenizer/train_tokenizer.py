import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(input_path: str, out_path: str, vocab_size: int) -> Tokenizer:
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
    )
    tok.train([input_path], trainer)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    return tok


def sanity_report(tok: Tokenizer, samples: list[str]) -> dict:
    report = {"vocab_size": tok.get_vocab_size(), "samples": []}
    for s in samples:
        ids = tok.encode(s).ids
        ok = tok.decode(ids) == s
        report["samples"].append({"text": s, "n_tokens": len(ids), "roundtrip_ok": ok})
    report["all_roundtrip_ok"] = all(s["roundtrip_ok"] for s in report["samples"])
    return report


def main():
    p = argparse.ArgumentParser(description="Train byte-level BPE tokenizer")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vocab-size", type=int, default=49152)
    args = p.parse_args()

    tok = train_tokenizer(args.input, args.out, args.vocab_size)
    samples = [
        "AAPL closed at 187.43 (+1.2%) on 2026-08-21.",
        "The NIFTY50 index fell 245 points amid rate concerns.",
        "Revenue was ₹1,234 crore vs ₹987 crore last quarter.",
        "def sharpe_ratio(returns): return returns.mean() / returns.std()",
    ]
    report = sanity_report(tok, samples)
    report_path = Path(args.out).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"vocab_size={report['vocab_size']} saved={args.out}")
    print(f"all_roundtrip_ok={report['all_roundtrip_ok']}")
    if not report["all_roundtrip_ok"]:
        raise SystemExit("tokenizer round-trip failed")


if __name__ == "__main__":
    main()
