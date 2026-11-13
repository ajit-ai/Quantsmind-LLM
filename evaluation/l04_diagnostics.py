"""TAKSH L0.4 — reproducible code-generation failure diagnostics.

Why does the L0.3 checkpoint produce fluent English instead of Python? Every
measurement here is re-derivable from the artifacts in this repository.

Usage (from the repository root):

    python -m evaluation.l04_diagnostics --out evaluation/l04_diagnostic_report.json

Sections:
  tokens      Python vs English coverage by the shipped tokenizer
  indent      indentation representation (4-space, 8-space, tab)
  identifiers identifier fragmentation in real code
  docs        code-document length, truncation vs seq_len, syntax parsability
  separators  <eos> usage vs document count
  mix         code vs general-text token distribution in the L0.3 corpus
  model       next-token distribution of the L0.3 checkpoint on code vs English

Everything is deterministic. Nothing trains. `--skip-model` skips the (slow)
model probes.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TOK_PATH = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
L03_TRAIN = REPO / "data" / "l03_corpus" / "train.jsonl"

PYTHON_KEYWORDS = [
    "def", "return", "if", "elif", "else", "for", "while", "class", "import",
    "from", "as", "with", "try", "except", "finally", "raise", "assert",
    "lambda", "pass", "break", "continue", "yield", "global", "nonlocal",
    "del", "in", "is", "not", "and", "or", "None", "True", "False",
]
PYTHON_BUILTINS = [
    "self", "print", "len", "range", "enumerate", "zip", "dict", "list", "set",
    "tuple", "str", "int", "float", "bool", "open", "sum", "min", "max", "abs",
    "sorted", "isinstance", "super", "type", "int", "input",
]
ENGLISH_WORDS = [
    "the", "of", "and", "to", "in", "is", "that", "for", "on", "as",
    "with", "they", "at", "be", "this", "have", "from", "or", "an", "by",
    "not", "are", "but", "what", "all", "were", "we", "when", "your", "can",
    "there", "use", "each", "which", "she", "how", "their", "if", "will",
    "up", "other", "about", "out", "many", "then", "them", "these", "so",
]

OPERATORS_MULTI = ["==", "!=", "<=", ">=", "->", "**", "//", "<<", ">>", "+=", "-=", "*=", "/=", ":="]
OPERATORS_SINGLE = list("()[]{}:,.+-*/%=@&|^<>!")


def _tok():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOK_PATH))


def _ids(tokenizer, text):
    return tokenizer.encode(text).ids


def measure_token_inventory(tokenizer):
    keywords = {k: len(_ids(tokenizer, k)) for k in PYTHON_KEYWORDS}
    builtins = {k: len(_ids(tokenizer, k)) for k in PYTHON_BUILTINS}
    english = {w: len(_ids(tokenizer, w)) for w in ENGLISH_WORDS}
    multi = {op: len(_ids(tokenizer, op)) for op in OPERATORS_MULTI}
    single = {op: len(_ids(tokenizer, op)) for op in OPERATORS_SINGLE}
    code_like_line = len(_ids(tokenizer, "def foo(x, y):\n    return x + y"))
    flat_line = len(_ids(tokenizer, "the quick brown fox jumps over the lazy dog"))
    return {
        "python_keywords": keywords,
        "python_builtins": builtins,
        "english_words": english,
        "multi_char_operators": multi,
        "single_char_operators": single,
        "single_token_rate": {
            "python_keywords": round(sum(v == 1 for v in keywords.values()) / len(keywords), 4),
            "python_builtins": round(sum(v == 1 for v in builtins.values()) / len(builtins), 4),
            "english_words": round(sum(v == 1 for v in english.values()) / len(english), 4),
            "multi_char_operators": round(sum(v == 1 for v in multi.values()) / len(multi), 4),
            "single_char_operators": round(sum(v == 1 for v in single.values()) / len(single), 4),
        },
        "sample_code_line_tokens": code_like_line,
        "sample_english_line_tokens": flat_line,
    }


def measure_indentation(tokenizer):
    out = {}
    for label, text in [("4 spaces", "    "), ("8 spaces", "        "), ("tab", "\t"),
                        ("4sp+x", "    x"), ("4sp+if", "    if")]:
        ids = _ids(tokenizer, text)
        out[label] = {"ids": ids, "n_tokens": len(ids),
                      "pieces": [tokenizer.decode([i]) for i in ids]}
    return out


IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def measure_identifiers(tokenizer, code_docs):
    lengths = []
    pieces = []
    for doc in code_docs:
        for name in IDENT_RE.findall(doc["text"]):
            if name in PYTHON_KEYWORDS:
                continue
            if len(name) < 2:
                continue
            lengths.append(len(name))
            pieces.append(len(_ids(tokenizer, name)))
    if not pieces:
        return {"count": 0}
    avg_len = sum(lengths) / len(lengths)
    avg_pieces = sum(pieces) / len(pieces)
    return {
        "identifiers_sampled": len(pieces),
        "avg_identifier_chars": round(avg_len, 3),
        "avg_tokens_per_identifier": round(avg_pieces, 3),
        "fraction_single_token": round(sum(p == 1 for p in pieces) / len(pieces), 4),
        "fraction_three_plus_token": round(sum(p >= 3 for p in pieces) / len(pieces), 4),
    }


def measure_documents(tokenizer, code_docs):
    lengths = []
    token_lens = []
    parse_ok = 0
    parse_total = 0
    truffle_512 = 0
    truffle_2048 = 0
    for doc in code_docs:
        text = doc["text"]
        ids = _ids(tokenizer, text)
        lengths.append(len(text))
        token_lens.append(len(ids))
        if len(ids) > 512:
            truffle_512 += 1
        if len(ids) > 2048:
            truffle_2048 += 1
        if doc.get("source", "").endswith(".py") or doc.get("source", "").startswith("m3-verified:"):
            parse_total += 1
            try:
                ast.parse(text)
                parse_ok += 1
            except Exception:
                pass
    n = len(lengths)
    token_lens_sorted = sorted(token_lens)
    return {
        "code_documents": n,
        "chars": {"min": min(lengths), "median": token_lens[n // 2] if False else sorted(lengths)[n // 2],
                  "max": max(lengths), "mean": round(sum(lengths) / n, 1)},
        "tokens": {"min": min(token_lens), "median": token_lens_sorted[n // 2],
                   "p90": token_lens_sorted[int(n * 0.9)],
                   "max": max(token_lens), "mean": round(sum(token_lens) / n, 1)},
        "documents_over_512_tokens": truffle_512,
        "documents_over_2048_tokens": truffle_2048,
        "python_parseable_docs": f"{parse_ok}/{parse_total}",
    }


def measure_separators(bin_path):
    import numpy as np

    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    eos = int((arr == 2).sum())
    return {
        "tokens_total": int(len(arr)),
        "eos_count": eos,
        "eos_fraction": round(eos / len(arr), 6),
        "eos_per_document": None,  # filled by caller with manifest doc counts
        "first_id": int(arr[0]),
        "last_id": int(arr[-1]),
    }


def measure_mix(records):
    domain_docs = collections.Counter()
    domain_chars = collections.Counter()
    for r in records:
        domain_docs[r["domain"]] += 1
        domain_chars[r["domain"]] += len(r["text"])
    total_docs = sum(domain_docs.values())
    total_chars = sum(domain_chars.values())
    return {
        "documents": dict(domain_docs),
        "chars": dict(domain_chars),
        "document_share": {k: round(v / total_docs, 4) for k, v in domain_docs.items()},
        "char_share": {k: round(v / total_chars, 4) for k, v in domain_chars.items()},
    }


CODE_PROBES = [
    ("code_eval", "def compute_total(items):\n    total = 0\n    for item in items:\n        total += item\n    return total"),
    ("code_def", "def add(a, b):\n    return a + b"),
]
ENGLISH_PROBES = [
    ("english_general", "The history of the world is a long and complicated story that"),
    ("english_econ", "Adam Smith argued that free markets allocate resources efficiently because"),
]


def measure_model(checkpoint_path):
    import torch
    import torch.nn.functional as F

    from serving.local_inference import load_model
    from tokenizers import Tokenizer

    device = "cpu"
    model, _ = load_model(checkpoint_path, device)
    tokenizer = Tokenizer.from_file(str(TOK_PATH))
    out = {}
    for label, probe in CODE_PROBES + ENGLISH_PROBES:
        ids = tokenizer.encode(probe).ids
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(x)
        probs = F.softmax(logits[0, -1].float(), dim=-1)
        top = torch.topk(probs, 8)
        entropy = float(-(probs * (probs + 1e-12).log()).sum())
        out[label] = {
            "prompt_tokens": len(ids),
            "next_token_entropy_nats": round(entropy, 4),
            "top1_prob": round(float(top.values[0]), 4),
            "top8": [{"token": tokenizer.decode([int(i)]), "p": round(float(p), 4)}
                     for p, i in zip(top.values, top.indices)],
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="evaluation/l04_diagnostic_report.json")
    parser.add_argument("--checkpoint", default="runs/l03_pretrain/checkpoint_last.pt")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args(argv)

    tokenizer = _tok()
    report = {}

    report["tokenizer"] = {
        "path": str(TOK_PATH.relative_to(REPO)),
        "vocab_size": tokenizer.get_vocab_size(),
        "specials": {m: tokenizer.token_to_id(m) for m in ("<pad>", "<bos>", "<eos>", "<unk>")},
    }
    report["token_inventory"] = measure_token_inventory(tokenizer)
    report["indentation"] = measure_indentation(tokenizer)

    with L03_TRAIN.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    code_docs = [r for r in records if r["domain"] == "code"]
    general_docs = [r for r in records if r["domain"] == "general"]
    report["mix_l03_train"] = measure_mix(records)
    report["identifiers_l03_code"] = measure_identifiers(tokenizer, code_docs)
    report["documents_l03_code"] = measure_documents(tokenizer, code_docs)

    sep = measure_separators(REPO / "data" / "l03_corpus" / "train.tokens.bin")
    manifest = json.loads((L03_TRAIN.parent / "dataset_manifest.json").read_text(encoding="utf-8"))
    sep["eos_per_document"] = round(sep["eos_count"] / manifest["train_documents"], 4)
    report["separators_l03_train"] = sep

    # Code classification of the training corpus: complete vs broken Python
    py_ok = 0
    py_total = 0
    for doc in code_docs:
        py_total += 1
        try:
            ast.parse(doc["text"])
            py_ok += 1
        except Exception:
            pass
    report["code_syntax_parse"] = {"parseable": py_ok, "total": py_total}

    if not args.skip_model:
        report["model_probes"] = measure_model(REPO / args.checkpoint)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k in ("tokenizer", "mix_l03_train", "separators_l03_train", "code_syntax_parse")}, indent=2, ensure_ascii=False))
    print(f"full report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
