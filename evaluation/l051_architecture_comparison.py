"""TAKSH L0.5.1 — multilingual architecture comparison (WP5).

Compares four multilingualization strategies for the same 28.8M-class Taksh
architecture:

  A  one shared multilingual model
  B  shared foundation + language-specific adapters (LoRA-class)
  C  shared foundation + language-family adapters
  D  separate per-language models (fallback comparison only)

Scores are engineering judgement anchored to measured facts in this repository
(28.8M params, 0/30 pass@1, pipeline verified, data: 1.59M Python tokens,
zero for every other L1.0 language). No model is created or evaluated here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l05_architecture_audit import count_parameters  # noqa: E402

BASE_CFG = {"vocab_size": 5822, "d_model": 384, "n_layers": 12, "n_heads": 8,
            "n_kv_heads": 4, "ffn_dim": 1536, "max_seq_len": 2048,
            "rope_theta": 10000.0, "norm_eps": 1e-5, "tie_embeddings": True}

ADAPTER_RANK = 8


def adapter_params(rank: int = ADAPTER_RANK) -> int:
    """LoRA-class adapter per layer: low-rank for wq, wk, wv, wo, and SwiGLU trio.

    For each rank-r matrix pair (A: d_in x r, B: r x d_out), params = d_in*r + r*d_out.
    Applied to q/k/v/o projections and gate/up/down FFN matrices.
    """
    d = 384
    matrices = [
        (d, d),   # wq
        (d, 192),  # wk (4 KV heads x 48 head_dim)
        (d, 192),  # wv
        (d, d),   # wo
        (d, 1536),  # gate
        (d, 1536),  # up
        (1536, d),  # down
    ]
    per_layer = sum(d_in * rank + rank * d_out for d_in, d_out in matrices)
    return per_layer * 12


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l051_architecture_comparison.json")
    args = parser.parse_args(argv)

    base = count_parameters(BASE_CFG)["total_params"]
    adapter = adapter_params()
    n_lang = 16
    n_family = 8  # c_family, commercial, scripting, fp_family, apple_ml, scientific, sql, other

    comparison = {
        "schema": "taksh.l051_architecture_comparison/1",
        "generated_by": "evaluation/l051_architecture_comparison.py",
        "basis": {
            "base_params_measured": base,
            "adapter_rank": ADAPTER_RANK,
            "adapter_params_per_language": adapter,
            "n_l1_languages": n_lang,
            "n_family_groups": n_family,
            "measured_anchors": [
                "L0.2: 28.8M model, 260-example SFT, 0/30 unseen pass@1 (capability ~zero)",
                "L0.3: continued pretraining on 20M tokens -> code-val 6.87 -> 5.41, still 0/30",
                "L0.4: code-dense warmstart -> code-val 5.06-5.51, still 0/30",
            ],
        },
        "options": [
            {
                "id": "A",
                "name": "One shared multilingual model",
                "params_total": base,
                "checkpoint_storage_bytes": base * 4,
                "inference_memory_fp32_bytes": base * 4,
                "training_complexity": "lowest: single run, single corpus mixture; no adapter bookkeeping",
                "maintainability": "excellent: one artifact, one tokenizer, one checkpoint lineage",
                "cross_language_transfer": "positive by construction (shared parameters learn common structure); strongest for syntactically close languages (C/C++/C#/Java/Go/Rust)",
                "language_interference": "real: one shared distribution must serve all 16; weaker languages risk dilution (documented in multilingual literature)",
                "local_deployment": "best: one 115 MB fp32 checkpoint; CPU inference feasible (verified, L0.1)",
                "checkpoint_compatibility": "direct: existing v180_sft / l03_pretrain / l04 checkpoints are drop-in continuation targets",
                "ops_complexity": "lowest",
                "evidence_for": "only approach compatible with zero new architecture work; matches existing verified inference stack",
                "evidence_against": "unproven at 28.8M; interference unmeasured (no multilingual benchmark exists yet)",
                "verdict": "PREFERRED HYPOTHESIS (not proven); consistent with all current evidence",
            },
            {
                "id": "B",
                "name": "Shared foundation + language-specific adapters",
                "params_total": base + n_lang * adapter,
                "adapter_params_total": n_lang * adapter,
                "checkpoint_storage_bytes": base * 4 + n_lang * adapter * 4,
                "inference_memory_fp32_bytes": base * 4 + adapter * 4,
                "training_complexity": "moderate: shared pretraining + per-language SFT; 16 adapter lifecycles to version",
                "maintainability": "moderate: 17 artifacts (1 base + 16 adapters); rollback per adapter possible",
                "cross_language_transfer": "partial: shared trunk transfers; adapters memorize language-specific surface forms",
                "language_interference": "low: adapters isolate language-specific directions; dilution mostly avoided",
                "local_deployment": "good: base + 1/16th adapter per language; adapter load adds ~1.2 MB",
                "checkpoint_compatibility": "base reusable; adapters trained from scratch (no existing adapters)",
                "ops_complexity": "medium: adapter registry, routing/selection at inference, per-language eval",
                "evidence_for": "adapters are proven for parameter-efficient specialisation; inference cost barely changes",
                "evidence_against": "routing logic needed at inference (language detection); adapters cannot fix a weak shared trunk (we have one)",
                "verdict": "FALLBACK IF A's shared trunk shows destructive interference at scale; NOT FIRST CHOICE today",
            },
            {
                "id": "C",
                "name": "Shared foundation + language-family adapters",
                "params_total": base + n_family * adapter,
                "adapter_params_total": n_family * adapter,
                "checkpoint_storage_bytes": base * 4 + n_family * adapter * 4,
                "inference_memory_fp32_bytes": base * 4 + adapter * 4,
                "training_complexity": "moderate: family assignment decisions (which language goes where); 8 adapters",
                "maintainability": "good: 9 artifacts; coarser specialisation than B",
                "cross_language_transfer": "high within family (C/C++/C#/Java transfer strongly); zero across distant families except through trunk",
                "language_interference": "low-medium: family adapters absorb broad regularities; individual languages still diluted within family",
                "local_deployment": "good: base + one family adapter; routing needs family detection",
                "checkpoint_compatibility": "base reusable; family adapters trained from scratch",
                "ops_complexity": "medium-low",
                "evidence_for": "family granularity matches known software ecosystems (Curly-brace C-family vs scripting vs FP vs SQL)",
                "evidence_against": "family assignment is judgement, not measurement; no family-annotated corpus exists here yet",
                "verdict": "VIABLE VARIANT OF B; useful if B's 16 adapters are operationally heavy; not proven either",
            },
            {
                "id": "D",
                "name": "Separate per-language models",
                "params_total": base * n_lang,
                "checkpoint_storage_bytes": base * n_lang * 4,
                "inference_memory_fp32_bytes": base * 4,
                "training_complexity": "highest: 16 independent runs, 16 data pools, 16 eval cycles",
                "maintainability": "poor: 16 model lifecycles; no shared capability; regression per language invisible to others",
                "cross_language_transfer": "zero by construction",
                "language_interference": "none (each model isolated) — the only guaranteed isolation method, at maximal cost",
                "local_deployment": "poor: 1.84 GB total checkpoints; user must choose a model per language",
                "checkpoint_compatibility": "none shared; each trained separately; no reuse of this repo's verified pipeline gains",
                "ops_complexity": "highest by far",
                "evidence_for": "only valid when interference is proven severe at A scale AND per-language data suffices (none does today)",
                "evidence_against": "we cannot train even one competent model at 28.8M; 16 is absurd at current data/compute",
                "verdict": "REJECTED for now; keep as documented last-resort fallback",
            },
        ],
        "decision_matrix": {
            "dimensions": ["params_storage", "training_complexity", "inference_memory", "maintenance",
                           "cross_language_transfer", "interference_risk", "local_deployment",
                           "checkpoint_compatibility", "ops_complexity"],
            "summary": "A wins on storage/complexity/deployment; B/C win on isolation (unproven benefit today); D dominated on cost",
            "preference_order": ["A", "C", "B", "D"],
            "preference_caveat": "preference is an engineering hypothesis, not a measured result; A must be validated on the multilingual benchmark before lock-in",
        },
        "conclusions": [
            "do not redesign the architecture before the multilingual benchmark shows interference (unavailable data)",
            "A is the starting point; if A fails with clear interference signatures, fall back to C before B before D",
            "adapter strategies cannot be evaluated until a shared base demonstrates any capability at all (0/30 today)",
            "all four options are blocked on the same resource: licensed multilingual data at scale",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"adapter_params_per_language={adapter:,}")
    for opt in comparison["options"]:
        print(f"{opt['id']}: params={opt['params_total']:,}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
