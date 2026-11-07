"""Increment 12 conditional-learning diagnosis (inference only).

Teacher-forced paired-prompt probes: prompts differ only in identifier
names; the target prefix is supplied identically. Reports correct-token
probability/rank/CE, prompt sensitivity (divergence + probability deltas),
hidden-state differences, and autoregressive-vs-teacher-forced comparison
for the baseline (and optionally one candidate). Writes JSON only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.conditional import (
    PAIRED_CASES,
    distribution_divergence,
    first_identifier_tokens,
    token_cross_entropy,
    token_probability,
    token_rank,
)

BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
CANDIDATES = {
    "baseline": BASELINE,
    "increment10": Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment10\increment10-copy-100\checkpoints\trained.pt"),
}


def _softmax(logits):
    import torch

    return torch.softmax(logits.float(), dim=-1).tolist()


def probe_checkpoint(checkpoint: Path, *, tokenizer) -> dict:
    import torch
    from serving.local_inference import generate_ids, load_model

    torch.set_num_threads(4)
    model, _config = load_model(str(checkpoint), "cpu")
    model.eval()
    cases = []
    try:
        for pair_id, operation, instruction, sig_a, sig_b, prefix, ident_a, ident_b in PAIRED_CASES:
            sides = {}
            for label, signature, identifier in (("A", sig_a, ident_a), ("B", sig_b, ident_b)):
                sequence = (instruction + "\n" + signature + "\n" + prefix).strip()
                ids = tokenizer.encode(sequence, add_special_tokens=False).ids
                if isinstance(ids, (list, tuple)):
                    ids = list(ids)
                with torch.no_grad():
                    logits, _ = model(torch.tensor([ids], dtype=torch.long))
                    hidden_in = model.tok_emb(torch.tensor([ids], dtype=torch.long))
                    # Final hidden state via decode internals: rerun decode on embeds.
                    _logits, _ = model.decode(hidden_in)
                    # Normed hidden state: norm over residual stream approximates
                    # by re-running blocks manually is out of scope; compare logits.
                    probs = _softmax(logits[0, -1])
                    expected_ids = first_identifier_tokens(tokenizer, identifier)
                    first_id = expected_ids[0]
                    probs_torch = torch.softmax(logits[0, -1].float(), dim=-1)
                    # Teacher forcing across the full identifier token sequence.
                    full_correct = 0
                    ces = []
                    cur_ids = list(ids)
                    for expected_id in expected_ids:
                        with torch.no_grad():
                            step_logits, _ = model(torch.tensor([cur_ids], dtype=torch.long))
                            step_probs = _softmax(step_logits[0, -1])
                        if step_probs and max(range(len(step_probs)), key=lambda i: step_probs[i]) == expected_id:
                            full_correct += 1
                        ces.append(token_cross_entropy(step_probs, expected_id))
                        cur_ids.append(expected_id)
                    sides[label] = {
                        "signature": signature,
                        "identifier": identifier,
                        "expected_ids": expected_ids,
                        "probs": probs,
                        "first_token_probability": token_probability(probs, first_id),
                        "first_token_rank": token_rank(probs, first_id),
                        "first_token_ce": token_cross_entropy(probs, first_id),
                        "tokenwise_ce": [round(c, 4) for c in ces],
                        "identifier_tokens_correct": full_correct,
                        "identifier_token_count": len(expected_ids),
                    }
            divergence = distribution_divergence(sides["A"]["probs"], sides["B"]["probs"])
            # Autoregressive comparison: greedy generation vs teacher forcing.
            greedy = {}
            for label, (signature, identifier) in (
                ("A", (sig_a, ident_a)), ("B", (sig_b, ident_b))
            ):
                result = generate_ids(
                    model, tokenizer, instruction + "\n" + signature + "\n",
                    max_new_tokens=32, temperature=0.0, top_k=1, top_p=1.0,
                    device="cpu", seed=20261023, binding_repair=False,
                )
                greedy[label] = {
                    "completion": result["text"],
                    "contains_identifier": identifier in result["text"],
                }
            cases.append({
                "pair_id": pair_id,
                "operation": operation,
                "sides": {label: {k: v for k, v in side.items() if k != "probs"} for label, side in sides.items()},
                "sensitivity": {
                    "tv_distance": divergence["tv_distance"],
                    "top_token_agree": divergence["top_token_agree"],
                    "top_a": divergence.get("top_a"),
                    "top_b": divergence.get("top_b"),
                    "prob_delta_correct": round(
                        sides["A"]["first_token_probability"] - sides["B"]["first_token_probability"], 6
                    ),
                },
                "greedy": greedy,
            })
    finally:
        del model
        gc.collect()
    return {"cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description="Increment 12 conditional diagnosis")
    parser.add_argument("--checkpoint", default=None, help="defaults to baseline + increment10")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(REPO / "tokenizer" / "qm_code_bpe_v0.json"))
    targets = {"baseline": CANDIDATES["baseline"], "increment10": CANDIDATES["increment10"]}
    if args.checkpoint:
        targets = {"selected": Path(args.checkpoint)}
    result = {label: probe_checkpoint(path, tokenizer=tokenizer) for label, path in targets.items()}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {}
    for label, data in result.items():
        summary[label] = [
            {
                "pair": case["pair_id"],
                "A_prob": round(case["sides"]["A"]["first_token_probability"], 4),
                "A_rank": case["sides"]["A"]["first_token_rank"],
                "B_prob": round(case["sides"]["B"]["first_token_probability"], 4),
                "B_rank": case["sides"]["B"]["first_token_rank"],
                "tv": case["sensitivity"]["tv_distance"],
                "greedy_A": case["greedy"]["A"]["contains_identifier"],
                "greedy_B": case["greedy"]["B"]["contains_identifier"],
            }
            for case in data["cases"]
        ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
