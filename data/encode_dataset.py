import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description="Encode a text corpus to packed token ids")
    p.add_argument("--corpus", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=None)
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    ids: list[int] = []
    corpus = Path(args.corpus)
    with corpus.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.extend(tok.encode(line).ids)
            if args.max_tokens and len(ids) >= args.max_tokens:
                ids = ids[: args.max_tokens]
                break

    max_id = max(ids) if ids else 0
    dtype = np.uint16 if max_id < 65536 else np.uint32
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(ids, dtype=dtype).tofile(out)

    meta_path = out.with_suffix(".meta.json")
    meta = {
        "vocab_size": tok.get_vocab_size(),
        "num_tokens": len(ids),
        "dtype": np.dtype(dtype).name,
        "corpus": str(corpus),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"tokens={len(ids)} vocab={meta['vocab_size']} dtype={meta['dtype']} -> {out}")


if __name__ == "__main__":
    main()
