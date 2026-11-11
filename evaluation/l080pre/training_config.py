"""TAKSH L0.8.0-PRE — D: non-executing, versioned training configuration proposal.

Produces the canonical proposal artifact WITHOUT launching anything. Values
labelled measured / calculated / inherited / unverified-hypothesis. The owner
authorization field is intentionally fail-closed: NOT_AUTHORIZED while A1.1-C.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

FOUNDATION_SHA = "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"
TOK_COMMITTED = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"
SNAPSHOT = "l072-l04-fdb1fee2beaae9c2"


def build_proposal() -> dict:
    return {
        "schema": "taksh.l080pre_training_config/1",
        "proposal_id": "taksh-l080pre-training-proposal-v1",
        "version": 1,
        "created": "2026-11-11",
        "authorization": {
            "status": "NOT_AUTHORIZED",
            "owner_reference": "TAKSH-OWNER-DECISION-2026-11-11",
            "authorized_scope": "preparation only",
            "instruction": "do not launch training; register this file as a proposal only",
        },
        "model": {
            "config_id": "taksh-foundation-111m",
            "config_path": "configs/foundation/l060_foundation.yaml",
            "config_sha256": FOUNDATION_SHA,
            "architecture": "Llama-style decoder-only; vocab 5822; d_model 768; 12L; 12H; 6 KV; FFN 3072; RoPE 10000; RMSNorm; tied embeddings; context 2048; 110,658,816 params",
        },
        "tokenizer": {
            "artifact": "tokenizer/taksh_code_bpe_v1.json",
            "committed_blob_sha256": TOK_COMMITTED,
            "vocab_size": 5822,
        },
        "corpus": {
            "snapshot_id": SNAPSHOT,
            "snapshot_path": "artifacts/l072_l04_freeze/snapshot.json",
            "eligible_samples": 1050,
            "eligible_tokens": 1380611,
            "language": "python",
            "frozen_verdict": "FROZEN_RESEARCH_ONLY",
            "note": "single-language (Python) corpus; multilingual qualification is not claimed",
        },
        "sequence": {
            "context_length": 2048,
            "packing": "leftover-boundary packing with EOS separators; sample boundaries preserved (L0.7.1 pipeline artifacts)",
            "document_boundary_marker": "<eos> (id 2)",
            "max_tokens_per_batch": "4 micro-batch × 2048 context = 8,192 tokens (measured decision from L0.6.1)",
            "position_encoding": "RoPE",
        },
        "optimizer_candidates": [
            {"optimizer": "AdamW",
             "rationale": "proven for from-scratch code pretraining at this size; matches L0.6.1 canonical configuration (beta1 0.9, beta2 0.95, wd 0.1, clip 1.0)",
             "status": "inherited evidence"},
            {"optimizer": "Adafactor",
             "rationale": "factorized second moment; lower optimizer state for memory-constrained hosts; retuning needed",
             "status": "unverified hypothesis"},
        ],
        "lr_schedule_candidates": [
            {"schedule": "cosine with warmup",
             "params": "peak_lr 1e-3; warmup 200; min_lr_ratio 0.1",
             "status": "inherited evidence (L0.6.1 canonical configuration)"},
            {"schedule": "linear decay",
             "params": "peak_lr 1e-3; warmup 200",
             "status": "unverified hypothesis"},
        ],
        "batching": {
            "micro_batch_size": 4,
            "grad_accum_steps": 8,
            "tokens_per_step": 65536,
            "label": "calculated estimate; not validated end-to-end on the canonical model",
        },
        "checkpointing": {
            "every_steps": 500,
            "atomic_write": True,
            "integrity_required": True,
            "resume_must_validate": ["schema", "config fingerprint", "tokenizer fingerprint", "step monotonicity"],
        },
        "determinism": {
            "seed": 42,
            "deterministic_kernels": True,
            "note": "reproducibility requires fixed seed; bit-exact cross-hardware identity not claimed",
        },
        "validation": {
            "every_steps": 100,
            "val_batches": 8,
            "early_stopping": {"mode": "loss-plateau + nonfinite-guard", "status": "unverified hypothesis"},
        },
        "logging": {
            "telemetry": ["loss", "grad-norm", "lr", "step", "epoch", "tokens_processed",
                          "memory_rss", "checkpoint_event", "interruption"],
            "structured_report": True,
        },
        "recovery": {
            "resume_validation_gate": "mandatory (L0.6.1 gates required before resume)",
            "checkpoint_compatibility": "verified at load (fingerprint + tokenizer + schema)",
            "backwards_compatibility": "v180_sft is INCOMPATIBLE (GEOMETRY_INCOMPATIBLE) — never used",
        },
        "provenance": {
            "training_code_revision": "to be recorded at run time",
            "corpus_snapshot_binding": SNAPSHOT,
            "dataset_manifest_sha256": "recorded at run time from the L0.6.1 pipeline",
            "owner_authorization_id": "REQUIRED AT LAUNCH — NOT PROVIDED",
        },
        "limits": {
            "max_training_steps": 3000,
            "checkpoint_every_steps": 500,
            "eval_every_steps": 100,
            "notes": "ceilings only; actual run unauthorized",
        },
        "unverified_assumptions": [
            "Adafactor retuning is unproven at this parameter count",
            "full-run memory footprint is estimated, not measured",
            "CPU throughput at seq2048 is unknown",
            "no prior benchmark sanity run exists for the 111M config",
        ],
        "labels": {"inherited": ["AdamW/β/WD/clip/schedule/warmup from L0.6.1 canonical configuration"],
                   "calculated_estimates": ["tokens_per_step; working set"],
                   "unverified_hypotheses": ["Adafactor; early stopping; CPU feasibility"],
                   "measured": ["corpus token counts (from l072 freeze verification)"]},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l080pre_training_config.json")
    args = parser.parse_args(argv)

    proposal = build_proposal()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    print(f"training configuration proposal written to {out_path}")
    print(f"authorization status: {proposal['authorization']['status']} (NOT_AUTHORIZED by design)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
