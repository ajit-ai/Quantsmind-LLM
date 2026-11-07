"""Taksh v1.6.0 — Build SFT dataset from simple_humaneval problems.

Reads evaluation/simple_humaneval.py and writes a JSONL instruction-tuning
dataset where each example is: (instruction=partial function, response=expected tail).

Output: data/v160_sft/sft.jsonl

Usage:
    python scripts/build_v160_sft_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path("data/v160_sft")
OUT_FILE = OUT_DIR / "sft.jsonl"

INSTRUCTION_TEMPLATE = (
    "Complete the following Python function. Return only the code that "
    "replaces the ellipsis or completes the return statement. "
    "Do not repeat the function signature.\n\n{prompt}"
)


def main() -> None:
    from evaluation.simple_humaneval import PROBLEMS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for p in PROBLEMS:
        # Extract the partial function body from the prompt
        # Prompt looks like: "def add(a, b):\n    return "
        prompt_lines = p.prompt.strip().split("\n")
        last_line = prompt_lines[-1].rstrip()

        # Build the instruction
        instruction = INSTRUCTION_TEMPLATE.format(prompt=p.prompt)

        records.append({
            "id": p.id,
            "task": p.category,
            "instruction": instruction,
            "input": "",
            "response": p.expected_tail.strip(),
        })

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} SFT examples to {OUT_FILE}")
    print(f"Categories: {sorted({r['task'] for r in records})}")
    for r in records[:5]:
        print(f"  {r['id']}: {r['instruction'][:80]!r} -> {r['response']!r}")


if __name__ == "__main__":
    main()
