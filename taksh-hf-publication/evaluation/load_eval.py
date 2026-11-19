import json
from pathlib import Path

VALID_CATEGORIES = {"concept", "arithmetic", "regulation", "convention"}
SEED = Path(__file__).resolve().parents[1] / "evaluation" / "india_finance" / "seed_v0.jsonl"


def load_eval_set(path: Path = SEED) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        missing = {"id", "category", "question", "reference", "requires_tool"} - set(row)
        if missing:
            raise ValueError(f"{path.name}: item {row.get('id')} missing {missing}")
        if row["category"] not in VALID_CATEGORIES:
            raise ValueError(f"{path.name}: bad category {row['category']}")
        items.append(row)
    return items
