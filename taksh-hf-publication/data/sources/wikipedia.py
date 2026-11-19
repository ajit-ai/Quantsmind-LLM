import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Stream Wikipedia to JSONL (free, CPU-only)")
    p.add_argument("--out", default="data/raw/wikipedia/wiki_en.jsonl")
    p.add_argument("--lang", default="en", choices=["en", "hi"])
    p.add_argument("--max-docs", type=int, default=20000)
    p.add_argument("--min-chars", type=int, default=500)
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Missing optional dependency. Install with:\n  pip install datasets\n"
            "(free download; runs on CPU; no account needed)"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "wikimedia/wikipedia", f"20231101.{args.lang}", split="train", streaming=True
    )
    kept = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            if kept >= args.max_docs:
                break
            text = str(row.get("text", "")).strip()
            if len(text) < args.min_chars:
                continue
            f.write(
                json.dumps(
                    {
                        "text": text,
                        "source": f"wikipedia-{args.lang}",
                        "license": "CC-BY-SA-3.0",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
    print(f"wrote {kept} docs to {out}")


if __name__ == "__main__":
    main()
