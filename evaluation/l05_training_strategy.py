"""TAKSH L0.5 — WP3: training-strategy comparison.

Four candidate strategies, scored on data and compute realism against the
measured evidence in this repository (L0.1–L0.4 runs). This module produces no
training and downloads nothing; it is an analytical aid, grounded in
evaluation/l04_* artifacts and run manifests.

Writes reports/l05_training_strategy.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCENARIOS_PATH = REPO / "configs" / "feasibility" / "l05_scenarios.yaml"


def _strategies():
    return [
        {
            "id": "A",
            "name": "Continued pretraining (code-dense)",
            "description": "Warmstart from runs/l03_pretrain/checkpoint_last.pt and continue on a code-dense corpus (L0.4: 50.0% code, 3.18M train tokens).",
            "measured_evidence": [
                "L0.3 warmstart run: train loss 8.40 -> 5.05 over 600 steps (1.23M tokens); code-val loss 6.87 -> 5.41 (ppl 967 -> 224).",
                "L0.4 E1 warmstart: code-val loss 5.84 -> 5.34 over 300 steps; E2 code-only 5.62 -> 5.06; E3 (half LR) 5.51.",
                "Every L0.4 checkpoint still scores 0/30 on the frozen Python benchmark.",
            ],
            "requirements": {
                "unique_data": ">= 10-50x the current 1.59M qualified code tokens to begin transferring; 1B-class would need ~100B tokens (off the table)",
                "compute": "full 20M-token epoch ~ 28 CPU-days; A100-class ~ 1-2 hours",
                "tokenizer": "unchanged",
            },
            "advantages": [
                "tokenizer and checkpoint lineage preserved",
                "cheapest capability path per dollar of new data",
                "monitored divergence (val loss) gives an early stopping signal",
            ],
            "limitations": [
                "retains English-dominant gravity: without code mass, gains stay numerical (perplexity), not functional",
                "no behavioral alignment: still cannot follow instructions or stop",
                "diminishing returns per epoch; ~4 epochs is a practical ceiling on 17k-line subsets",
            ],
            "risks": [
                "same English-dominance failure mode as L0.3 revisited at larger scale",
                "license-filtered pool too small to reach useful perplexity scales",
            ],
            "unknowns": [
                "transfer curve beyond ~2M code tokens (not measurable in this environment)",
            ],
            "verdict": "CONTINUE ONLY WITH NEW LICENSED DATA AT SCALE; not a fix in isolation",
            "missing_items": ["owner-acquired >= 10GB licence-verified Python", "GPU budget"],
        },
        {
            "id": "B",
            "name": "Code-focused supervised fine-tuning only",
            "description": "SFT on high-quality instruction->code pairs with tests and explanations (the v160/v180 SFT recipe, scaled).",
            "measured_evidence": [
                "L0.2 overfit gate: 24/24 memorization with 260 pairs + 1e-3 lr — proves the pipeline can memorize.",
                "L0.2 hp: 260 pairs at 1e-3, 500 steps: train loss 6.38 -> 0.39, code-val loss 4.03 -> ?; still 0/30 unseen.",
                "L0.2 SFT runs systematically degrade code modeling (v180 code val 7.60 vs base 7.43; l02_sft_hp code val 12.31 — catastrophic forgetting of even the tiny base ability).",
                "SFT data with expression-only targets (v1.6/v1.8.0) taught expression-fragment completions and destroyed stopping behavior.",
            ],
            "requirements": {
                "unique_data": ">= 100k-1M high-quality instruction pairs (with tests where feasible) to beat chance on unseen tasks; we have 260 (+ 610 license-verified instruct_v1)",
                "quality": "task descriptions, correct Python solutions, test cases, explanations or code-only targets; multi-step and debugging examples",
                "provenance": "hand-written or license-verified; no LLM-generated pairs (project policy); no benchmark-derived pairs",
            },
            "advantages": [
                "most direct route to instruction following and EOS behaviour",
                "each example can be executed and its answer verified before training",
            ],
            "limitations": [
                "does NOT teach Python itself; only the instruction->code mapping",
                "quality-per-example dominates volume; 57-260 examples cannot beat 0/30",
                "memorization masquerades as capability (24/24 gate fooled nobody on unseen tasks)",
            ],
            "risks": [
                "catastrophic forgetting of the (already tiny) code-modeling ability",
                "SFT distribution shift can actively worsen held-out code loss, as measured",
            ],
            "unknowns": [
                "the pair volume at which SFT alone yields nonzero pass@1 for this 28.8M base (untested; expected: far beyond available data)",
            ],
            "verdict": "NECESSARY BUT NOT SUFFICIENT; without a strengthened base it produces memorization, not capability",
            "missing_items": ["verified instruction-pair corpus >= 100k", "base with real code-modeling ability"],
        },
        {
            "id": "C",
            "name": "Combined: continued pretraining + code-focused SFT (two-phase)",
            "description": "Phase 1 continued code-dense pretraining (A), then Phase 2 code-focused SFT (B).",
            "measured_evidence": [
                "Phase 1 works: L0.3/L0.4 show monotone code-val improvement (5.96 -> 5.06) with finite gradients and stable clipping.",
                "Phase 2 works mechanically: L0.2 shows pipelines execute, verify, and save cleanly.",
                "No combination attempted yet can be expected to exceed 0/30: each phase is individually data-starved by ~100x.",
            ],
            "requirements": {
                "phase1": " >= 10B tokens of licence-verified code for a 100M-300M model; or accept sub-1B target with smaller but still enormous pool",
                "phase2": " >= 100k verified instruction-code pairs with tests",
                "compute": "phase 1 ~ GPU-hours-to-days depending on size; phase 2 ~ CPU-days once base exists",
            },
            "advantages": [
                "aligns with how working code models are actually built (pretrain+anneal+SFT)",
                "both phases are measurable independently (val loss; pass@1 on a frozen benchmark)",
            ],
            "limitations": [
                "still wholly gated by licensed code at scale; phase 1 cannot be substituted by cleverness",
                "SFT on a weak base cannot raise pass@1; phase order matters",
            ],
            "risks": [
                "SFT after continued PT could still erase the phase-1 gains if data licensing forces a tiny phase-2 set",
            ],
            "unknowns": [
                "interaction term between phase-1 volume and phase-2 pair quality for a from-scratch 100M model (no data, no measurement possible here)",
            ],
            "verdict": "THE VIABLE ROUTE; both phases required; both blocked on licensed data + compute authorization",
            "missing_items": ["owner data acquisition budget", "GPU budget", "legal review of candidate corpora"],
        },
        {
            "id": "D",
            "name": "Architecture redesign (larger model vs. changes)",
            "description": "Scale up width/depth within the existing Taksh family; or change modules (attention, norm, positional encoding).",
            "measured_evidence": [
                "Architecture is NOT the measured bottleneck: perplexity falls monotonically with code-dense training; failures are data-starvation, not attention/norm/rope.",
                "Mechanically verified: RMSNorm+RoPE+GQA+SwiGLU+tied-embedding counts exactly (28,787,328 analytic == measured) and trains stably (grad norms 0.7-2.9).",
                "RoPE validated only to 2048 context; no evidence any module-level change would help at this scale.",
            ],
            "requirements": {
                "if_scaling": "100M-300M configuration with identical module family (candidate A/B); vocab unchanged",
                "if_redesign": "a measured failure that module changes would fix — none identified; any redesign is speculative",
            },
            "advantages": [
                "zero architecture-risk: same family scales cleanly to A/B/C",
            ],
            "limitations": [
                "redesign without evidence adds unquantifiable risk and zero measured benefit",
                "larger models REQUIRE more data; scaling alone with 1.59M code tokens accelerates memorization",
            ],
            "risks": [
                "C_1B without a matched data increase is wasted memory, not capability",
            ],
            "unknowns": [
                "ablations (e.g. different GQA ratio, finer token granularity) untested; all blocked on data anyway",
            ],
            "verdict": "DO NOT REDESIGN; scale with data. Consider A/B only after data authorization",
            "missing_items": ["license-verified data volume (gate)", "GPU budget (gate)"],
        },
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l05_training_strategy.json")
    args = parser.parse_args(argv)

    strategies = _strategies()
    out = {
        "schema": "taksh.l05_training_strategy/1",
        "generated_by": "evaluation/l05_training_strategy.py",
        "selection_context": "L0.1-L0.4 measured: tokenizer adequate; 28.8M model; 20M-token licensed corpus available but 96% general text; frozen benchmark 0/30 for all checkpoints.",
        "strategies": strategies,
        "comparison_matrix": [
            {
                "id": s["id"],
                "data_blocker": s["verdict"],
                "requires_owner_data_spend": s["id"] in ("A", "C"),
                "requires_owner_gpu_spend": s["id"] in ("A", "C"),
                "compatible_with_tokenizer": True,
                "compatible_with_checkpoints": s["id"] in ("A",),
                "expected_effect_at_current_scale": "zero capability gain (0/30) at current data scale",
            }
            for s in strategies
        ],
        "recommendation": (
            "Sequence: (1) owner authorizes data acquisition + legal review (candidate corpora in the register); "
            "(2) A100-class compute; (3) strategy C: continued pretraining at 100M, then code-focused SFT; "
            "(4) re-run the frozen L0.2 benchmark with the identical protocol; "
            "(5) keep the L0.1 default checkpoint until a candidate demonstrably beats it on the frozen benchmark."
        ),
        "do_not": [
            "do not switch the L0.1 default checkpoint",
            "do not train on L0.2 benchmark items or derivatives",
            "do not use commercial-LLM-generated pairs",
            "do not scale the model beyond data scale",
            "do not migrate the tokenizer",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for s in strategies:
        print(f"strategy {s['id']}: {s['name']}")
        print(f"    verdict: {s['verdict']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
