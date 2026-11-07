import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.build_code_corpus import process_local_files, chunk_code, is_minified, has_valid_structure
from data.filters import ExactDedup, MinHashDedup, quality_check, scrub_pii

corpus_path = Path("data/code_corpus_v0.jsonl")
stats_path = Path("data/code_corpus_v0.stats.json")

dedup = ExactDedup()
fuzzy = MinHashDedup()
stats = {
    "seen": 0,
    "kept": 0,
    "rejected_quality": 0,
    "rejected_dup": 0,
    "rejected_fuzzy": 0,
    "rejected_minified": 0,
    "rejected_syntax": 0,
    "chunks_emitted": 0,
    "by_language": {},
    "by_source": {},
}

items = []
for root_dir in [".", "training", "serving", "evaluation", "tokenizer", "data", "scripts", "tests"]:
    p = Path(root_dir)
    if p.exists():
        items.extend(process_local_files(p))

print(f"Total items: {len(items)}")

with corpus_path.open("w", encoding="utf-8") as f:
    for text, lang, source in items:
        stats["seen"] += 1
        if is_minified(text):
            stats["rejected_minified"] += 1
            continue
        if not has_valid_structure(text, lang):
            stats["rejected_syntax"] += 1
            continue
        ok, _ = quality_check(text, 100, 100_000, 0.3)
        if not ok:
            stats["rejected_quality"] += 1
            continue
        for piece in chunk_code(text, 200):
            if not dedup.add(piece):
                stats["rejected_dup"] += 1
                continue
            if not fuzzy.add(piece):
                stats["rejected_fuzzy"] += 1
                continue
            clean = scrub_pii(piece)
            f.write(json.dumps({"text": clean, "language": lang, "source": source}, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats["chunks_emitted"] += 1
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1

stats_path.write_text(json.dumps(stats, indent=2))
print(f"seen={stats['seen']} kept={stats['kept']} chunks={stats['chunks_emitted']}")
print(f"by_language={stats['by_language']}")
