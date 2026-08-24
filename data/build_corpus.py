import argparse
import json
from pathlib import Path

from data.filters import ExactDedup, quality_check, scrub_pii


def iter_docs(raw_dir: Path):
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    yield str(row.get("text", "")), str(
                        row.get("source", path.parent.name)
                    ), str(row.get("license", "unknown"))
        elif path.suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            yield text, path.parent.name, "see-MANIFEST"


def chunk_text(text: str, chunk_chars: int, min_chars: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            nl = text.rfind("\n", start + int(chunk_chars * 0.5), end)
            if nl > start:
                end = nl + 1
        piece = text[start:end].strip()
        if len(piece) >= min_chars:
            chunks.append(piece)
        start = end
    return chunks


def main():
    p = argparse.ArgumentParser(description="Build versioned corpus v0 from data/raw")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/corpus_v0.jsonl")
    p.add_argument("--stats", default="data/corpus_v0.stats.json")
    p.add_argument("--chunk-chars", type=int, default=6000)
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dedup = ExactDedup()
    stats = {
        "seen": 0,
        "kept": 0,
        "rejected_quality": 0,
        "rejected_dup": 0,
        "chunks_emitted": 0,
        "by_source": {},
    }

    with out.open("w", encoding="utf-8") as f:
        for text, source, license_tag in iter_docs(raw_dir):
            stats["seen"] += 1
            for piece in chunk_text(text, args.chunk_chars, 200):
                ok, _reason = quality_check(piece)
                if not ok:
                    stats["rejected_quality"] += 1
                    continue
                if not dedup.add(piece):
                    stats["rejected_dup"] += 1
                    continue
                clean = scrub_pii(piece)
                f.write(
                    json.dumps(
                        {"text": clean, "source": source, "license": license_tag},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["kept"] += 1
                stats["chunks_emitted"] += 1
                stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

    Path(args.stats).write_text(json.dumps(stats, indent=2))
    print(
        f"seen={stats['seen']} kept={stats['kept']} "
        f"dup={stats['rejected_dup']} quality_reject={stats['rejected_quality']}"
    )


if __name__ == "__main__":
    main()
