"""TAKSH L0.7.1 — sequence packing and context preparation.

Greedy deterministic packing respecting the canonical 2,048-token context.
Oversized samples are REJECTED with accounting, never silently truncated.
Sample/document boundaries stay traceable via end-of-sample markers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CONTEXT_LIMIT = 2048
EOS_ID = 2  # canonical tokenizer: <eos> at id 2


def pack_sequences(token_seqs: list[dict], context_limit: int = CONTEXT_LIMIT,
                   eos_id: int = EOS_ID) -> dict:
    """Greedy first-fit in input order (deterministic).

    Each input item: {"sample_id": str, "tokens": [int]}. Oversized items
    (tokens + eos > context_limit) are rejected and accounted, not truncated.
    """
    packed: list[dict] = []
    current: dict | None = None
    rejected = []
    boundaries = []
    for item in token_seqs:
        tokens = list(item["tokens"]) + [eos_id]
        if len(tokens) > context_limit:
            rejected.append({"sample_id": item["sample_id"],
                             "token_count": len(tokens),
                             "reason": "exceeds_context_limit"})
            continue
        if current is None:
            current = {"ids": [], "members": []}
        if len(current["ids"]) + len(tokens) > context_limit:
            packed.append(finalize_pack(current, context_limit))
            current = {"ids": [], "members": []}
        current["ids"].extend(tokens)
        current["members"].append(item["sample_id"])
        boundaries.append({"sample_id": item["sample_id"],
                           "sequence_offset": len(packed),
                           "position": len(current["ids"]) - len(tokens)})
    if current:
        packed.append(finalize_pack(current, context_limit))


    return {
        "context_limit": context_limit,
        "eos_id": eos_id,
        "packed_sequences": len(packed),
        "total_tokens": sum(len(p["ids"]) for p in packed),
        "rejected_sequences": rejected,
        "rejected_count": len(rejected),
        "samples_packed": sum(len(p["members"]) for p in packed),
        "all_within_context": all(len(p["ids"]) <= context_limit for p in packed),
        "boundaries_traceable": len(boundaries) == (len(token_seqs) - len(rejected)),
        "pack_manifest_sha256": hashlib.sha256(b"".join(
            json.dumps({"members": p["members"], "n": len(p["ids"])}, sort_keys=True).encode()
            for p in packed)).hexdigest(),
        "boundaries": boundaries,
        "packed": packed,
    }


def finalize_pack(current: dict, context_limit: int) -> dict:
    ids = current["ids"]
    return {
        "ids": ids,
        "members": current["members"],
        "length": len(ids),
        "padding_tokens": context_limit - len(ids),
        "attention_mask_member_spans": len(current["members"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l071_sequence_prep.json")
    args = parser.parse_args(argv)

    fixtures = [
        {"sample_id": "syn-a", "tokens": list(range(100))},
        {"sample_id": "syn-b", "tokens": list(range(300))},
        {"sample_id": "syn-c", "tokens": list(range(2000))},   # oversized
        {"sample_id": "syn-d", "tokens": list(range(1500))},
        {"sample_id": "syn-e", "tokens": list(range(400))},
    ]
    result = pack_sequences(fixtures)
    result.pop("packed")           # keep report compact; ids reproducible via fixtures
    result["fixtures_label"] = "synthetic"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "schema": "taksh.l071_sequence_prep/1",
        "generated_by": "evaluation/l071_packing.py",
        **result,
        "policy": "no truncation of source code; oversized samples rejected and accounted",
    }, indent=2), encoding="utf-8")
    print(f"packed {result['samples_packed']} samples into {result['packed_sequences']} sequences; rejected={result['rejected_count']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
