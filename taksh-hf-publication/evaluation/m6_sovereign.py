"""M6 Track B: sovereign-model research assessment.

M6 does **not** train anything. This module produces the engineering decision
plan for a future controlled from-scratch experiment: what data exists, what
data is missing, what each candidate scale would cost, and the gates that must
be satisfied before any such run is authorised.

Every cost figure is an ESTIMATE produced by ``training.preflight``, the same
estimator the training pipeline gates itself on, so M6 cannot drift from the
project's own notion of memory. Nothing here is a measured training result.

    python -m evaluation.m6_sovereign
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

MIB = 1024 ** 2

# Task coverage the sovereign track is expected to serve, paired with the
# benchmark categories M6 can actually score.
TASK_COVERAGE = [
    ("code_completion", "PARTIAL",
     "the corpus is Python library source, so completion-style supervision "
     "exists implicitly through next-token prediction"),
    ("code_generation", "ABSENT",
     "no instruction->code pairs; the corpus is raw source, not prompts"),
    ("debugging", "ABSENT", "no failing-code/patch pairs anywhere in the corpus"),
    ("refactoring", "ABSENT", "no before/after refactor pairs"),
    ("unit_test_generation", "ABSENT",
     "tests appear as source text, never as instruction->test pairs"),
    ("code_explanation", "ABSENT", "no natural-language explanation targets"),
    ("documentation", "ABSENT", "docstrings exist inside files but no "
                                "docstring-generation supervision"),
    ("sql", "ABSENT", "the verified corpus contains no SQL"),
    ("architecture", "ABSENT", "no design or architecture material"),
    ("devops_infrastructure", "ABSENT", "no CI, Docker or infra configuration"),
    ("instruction_following", "ABSENT",
     "absent by construction; M4 attempted to synthesise it from code and "
     "produced an instruction/completion mismatch"),
    ("structured_output", "ABSENT", "no schema-constrained examples"),
]

# Candidate scales for the next controlled experiment. Architecture values are
# scaled from the existing Llama-style decoder so the estimator applies
# unchanged. These are proposals, not commitments.
SCALE_PROPOSALS = [
    {"id": "A-baseline", "name": "Existing 2.85M research baseline",
     "d_model": 160, "n_layers": 5, "n_heads": 4, "n_kv_heads": 2,
     "ffn_dim": 640,
     "purpose": "reproducibility baseline; already trained, keep unchanged"},
    {"id": "B-existing-9m", "name": "Existing 9.36M config",
     "d_model": 256, "n_layers": 8, "n_heads": 8, "n_kv_heads": 4,
     "ffn_dim": 1024,
     "purpose": "already authored and preflight-measured; untrained"},
    {"id": "C-modest-30m", "name": "Modest 30M proposal",
     "d_model": 384, "n_layers": 10, "n_heads": 6, "n_kv_heads": 2,
     "ffn_dim": 1536,
     "purpose": "smallest scale that could plausibly generalise across tasks"},
    {"id": "D-125m", "name": "125M proposal",
     "d_model": 768, "n_layers": 12, "n_heads": 12, "n_kv_heads": 4,
     "ffn_dim": 2048,
     "purpose": "GPT-2-small class; requires compute this host does not have"},
    {"id": "E-350m", "name": "350M proposal",
     "d_model": 1024, "n_layers": 24, "n_heads": 16, "n_kv_heads": 4,
     "ffn_dim": 2816,
     "purpose": "only defensible with a GPU and a matured dataset"},
]

REQUIRED_TRAINING_TOKENS_PER_PARAMETER = 20  # Chinchilla-style lower bound


@dataclass
class ScaleEstimate:
    scale_id: str
    name: str
    parameters: int | None = None
    estimate_bytes: int | None = None
    tokens_for_20x: int | None = None
    fits_host: bool | None = None
    fits_total_ram: bool | None = None
    fits_available_ram: bool | None = None
    measured_peak_bytes: int | None = None
    note: str = ""
    status: str = "ESTIMATE"

    def to_dict(self) -> dict:
        data = {
            "scale_id": self.scale_id, "name": self.name,
            "parameters": self.parameters,
            "estimated_steady_state_mib": (round(self.estimate_bytes / MIB, 1)
                                           if self.estimate_bytes else None),
            "tokens_for_20x_parameters": self.tokens_for_20x,
            "fits_total_ram": self.fits_total_ram,
            "fits_available_ram": self.fits_available_ram,
            "fits_host": self.fits_host,
            "measured_peak_mib": (round(self.measured_peak_bytes / MIB, 1)
                                  if self.measured_peak_bytes else None),
            "note": self.note, "status": self.status,
        }
        return data


# Measured in configs/train/pretrain_m3.yaml on this host, not re-derived here.
MEASURED_REFERENCE = {
    "A-baseline": None,
    "B-existing-9m": 207 * MIB,
}


def _model_config(proposal) -> dict:
    return {
        "name": proposal["id"], "vocab_size": 5822,
        "d_model": proposal["d_model"], "n_layers": proposal["n_layers"],
        "n_heads": proposal["n_heads"], "n_kv_heads": proposal["n_kv_heads"],
        "ffn_dim": proposal["ffn_dim"], "max_seq_len": 512,
        "rope_theta": 10000.0, "norm_eps": 1.0e-5, "tie_embeddings": True,
    }


def _train_config() -> dict:
    return {
        "data": {"seq_len": 512},
        "batch": {"micro_batch_size": 1, "grad_accum_steps": 8},
        "optimizer": {"type": "adamw"},
        "precision": {"dtype": "float32"},
    }


@lru_cache(maxsize=None)
def _estimate_scale(scale_id: str) -> tuple[int, int]:
    """Parameter count and steady-state bytes for one proposal.

    ``training.preflight.estimate_memory`` walks the config and is slow (~5s per
    scale), while the answer depends only on the fixed proposal. Caching keeps
    repeated report builds cheap; RAM availability is still measured fresh by
    the caller.
    """
    proposal = next(p for p in SCALE_PROPOSALS if p["id"] == scale_id)
    from training.preflight import estimate_memory, model_size

    params = model_size(_model_config(proposal))["total_params"]
    memory = estimate_memory(_model_config(proposal), _train_config())
    return int(params), int(memory["total_bytes"])


def estimate_scales() -> list[ScaleEstimate]:
    """Estimate memory and data appetite for each proposed scale.

    Two verdicts are recorded, because they differ sharply on a shared machine:
    ``fits_total_ram`` compares against installed RAM, while
    ``fits_available_ram`` compares against what is actually free. A training
    run needs the second to be true, and this host is regularly near zero
    available.
    """
    from training.preflight import system_memory

    memory_info = system_memory()
    total_ram = memory_info.get("total_bytes") or 0
    available_ram = memory_info.get("available_bytes") or 0
    out = []
    for proposal in SCALE_PROPOSALS:
        est = ScaleEstimate(scale_id=proposal["id"], name=proposal["name"])
        try:
            params, estimate_bytes = _estimate_scale(proposal["id"])
            est.parameters = params
            est.estimate_bytes = estimate_bytes
            est.tokens_for_20x = params * REQUIRED_TRAINING_TOKENS_PER_PARAMETER
            est.fits_total_ram = bool(total_ram and estimate_bytes <= total_ram)
            est.fits_available_ram = bool(available_ram
                                          and estimate_bytes <= available_ram)
            est.fits_host = bool(est.fits_total_ram and est.fits_available_ram)
            est.measured_peak_bytes = MEASURED_REFERENCE.get(proposal["id"])
            est.note = proposal["purpose"]
        except Exception as exc:  # noqa: BLE001 - a failed estimate is reported
            est.status = "ESTIMATE_FAILED"
            est.note = f"could not estimate: {exc}"
        out.append(est)
    return out


def data_assessment() -> dict:
    """Data gaps, derived from the dataset manifest rather than asserted."""
    manifest_path = Path("data/m3_code_verified.manifest.json")
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}

    train = (manifest.get("splits") or {}).get("train") or {}
    val = (manifest.get("splits") or {}).get("val") or {}
    gate = manifest.get("licence_gate") or {}

    verified = gate.get("verified_sources") or []
    withheld = gate.get("withheld_sources") or {}

    return {
        "manifest_present": bool(manifest),
        "train_records": train.get("records"),
        "train_repositories": train.get("repositories"),
        "train_tokens": train.get("token_count"),
        "train_licenses": train.get("licenses"),
        "val_records": val.get("records"),
        "val_tokens": val.get("token_count"),
        "verified_sources": verified,
        "withheld_sources": withheld,
        "holdout": manifest.get("holdout"),
        "task_coverage": [
            {"task": task, "coverage": status, "basis": basis}
            for task, status, basis in TASK_COVERAGE],
        "coverage_summary": {
            "partial": sum(1 for _, s, _ in TASK_COVERAGE if s == "PARTIAL"),
            "absent": sum(1 for _, s, _ in TASK_COVERAGE if s == "ABSENT"),
            "total": len(TASK_COVERAGE),
        },
        "disablement_note": (
            "The verified corpus is Python library source only. It supports "
            "next-token prediction, which is why M3 reached a low loss, but it "
            "contains no supervision for debugging, refactoring, test "
            "generation, explanation, documentation, SQL or architecture. A "
            "low loss on this corpus cannot become coding capability on those "
            "tasks without new data."
        ),
    }


def build_assessment(run_manifests=None) -> dict:
    """The complete Track B assessment."""
    from evaluation.m6_resources import machine_facts

    scales = estimate_scales()
    data = data_assessment()

    largest_fitting = [s for s in scales if s.fits_host]
    return {
        "schema": "quantsmind.m6_sovereign/1",
        "generated_at": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "training_performed": False,
        "training_authorised": False,
        "machine": machine_facts(),
        "established": [
            "a working from-scratch pretraining and SFT pipeline exists and "
            "has produced loadable checkpoints",
            "the pipeline is licence-gated: only sources in "
            "licence_verification.json verified_ids enter training",
            "the CPU-only host can train a ~2.85M-parameter model, and a "
            "~9.36M-parameter configuration was preflight-measured at "
            "~207 MiB peak RSS",
            "the corpus comprises only licence-verified Python library source",
        ],
        "not_demonstrated": [
            "any useful coding capability (see the M6 capability report)",
            "functional correctness on any benchmark example",
            "instruction following above chance on the M5 benchmark",
            "that the corpus can teach any task beyond next-token prediction",
            "that a larger model would help, absent data for those tasks",
        ],
        "data": data,
        "scales": [s.to_dict() for s in scales],
        "compute_gap": {
            "total_ram_gib": machine_facts()["total_ram_gib"],
            "available_ram_gib": machine_facts()["available_ram_gib"],
            "cuda_available": machine_facts()["torch"].get("cuda_available"),
            "largest_proposal_fitting_host": (
                max(largest_fitting, key=lambda s: s.parameters).scale_id
                if largest_fitting else None),
            "note": (
                "Steady-state figures cover weights, gradients, AdamW moments "
                "and stored activations; they exclude tokenizer buffers and the "
                "OS. 'fits_available_ram' is the binding test on a shared "
                "machine: this host reports only a fraction of a GiB free, so "
                "even proposals that fit installed RAM cannot start while other "
                "work is running. Any scale beyond the largest fitting proposal "
                "needs hardware this host does not have."),
        },
        "validation_gates": [
            "G1 data: each target task has licence-verified supervision with "
            "an isolated holdout that shares no repository or content hash",
            "G2 preflight: training.preflight passes with a measured probe, "
            "not an estimate, on the actual training host",
            "G3 evaluation: a frozen benchmark exists BEFORE training starts "
            "and covers the tasks being claimed",
            "G4 capability: functional pass rate rises above the M3/M4 "
            "baseline on a benchmark large enough to be non-anecdotal",
            "G5 regression: general-code behaviour does not degrade",
        ],
        "recommended_next_experiment": {
            "scale": "B-existing-9m",
            "rationale": (
                "It is the largest configuration already authored and "
                "preflight-measured on this host, so it isolates the data "
                "question from the compute question. Scaling the model before "
                "fixing the data would repeat the M4 error of adding capacity "
                "without adding supervision."),
            "preconditions": [
                "licence-verified instruction data for at least "
                "code_generation, debugging and unit_test_generation",
                "a frozen benchmark covering those tasks, versioned "
                "separately from the M5 benchmark",
                "preflight re-run on the exact training host",
            ],
            "out_of_scope": [
                "any scale above the largest host-fitting proposal",
                "instruction tuning before pretraining data exists",
                "reusing the M4 synthetic instruction construction unchanged",
            ],
        },
        "authorisation_required": [
            "explicit human authorisation to start a training run",
            "confirmation that new data has passed the licence gate",
            "confirmation that a frozen benchmark predates the run",
        ],
    }


def format_assessment(report) -> str:
    lines = []
    add = lines.append
    add("M6 sovereign-track assessment (Track B)")
    add(f"  training performed : {report['training_performed']}")
    add(f"  training authorised: {report['training_authorised']}")
    data = report["data"]
    add("")
    add("  Data")
    add(f"    train: {data['train_records']} records / "
        f"{data['train_tokens']} tokens / "
        f"{len(data['train_repositories'] or [])} repositories")
    add(f"    val  : {data['val_records']} records / {data['val_tokens']} tokens")
    add(f"    withheld: {data['withheld_sources']}")
    add("")
    add("  Task coverage")
    for row in data["task_coverage"]:
        add(f"    {row['coverage']:8} {row['task']}")
    add("")
    add("  Scales (all ESTIMATE unless a measured probe is recorded)")
    for scale in report["scales"]:
        est = scale["estimated_steady_state_mib"]
        add(f"    {scale['scale_id']:16} params={scale['parameters']:>10,} "
            f"est={est} MiB total_ram={scale['fits_total_ram']} "
            f"available_ram={scale['fits_available_ram']}")
    add("")
    add(f"  Recommended next experiment: "
        f"{report['recommended_next_experiment']['scale']}")
    for precondition in report["recommended_next_experiment"]["preconditions"]:
        add(f"    - {precondition}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="M6 sovereign-model research assessment (no training)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_assessment()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2) if args.json
          else format_assessment(report))
    if args.out:
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())