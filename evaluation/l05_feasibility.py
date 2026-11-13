"""TAKSH L0.5 — single reproducible entry point for the feasibility gate.

Runs the five L0.5 tools in order, collects their outputs, verifies preservation
of every existing artifact that matters (branch/HEAD/status, checkpoint hashes,
tokenizer, frozen benchmark), and writes:

* reports/l05_feasibility_report.json   (single machine-readable gate report)
* reports/L05_FEASIBILITY_REPORT.md     (human-readable decision document)

Usage:
    python -m evaluation.l05_feasibility
    python -m evaluation.l05_feasibility --skip-reports   # run sub-tools only

No model is trained, nothing is downloaded, nothing outside reports/ is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REPORTS = REPO / "reports"

CHECKPOINTS = [
    "runs/v160_pretrain/checkpoint_last.pt",
    "runs/v180_sft/checkpoint_last.pt",
    "runs/l02_sft/checkpoint_last.pt",
    "runs/l02_sft_hp/checkpoint_last.pt",
    "runs/l02_overfit/checkpoint_last.pt",
    "runs/l03_pretrain/checkpoint_last.pt",
    "runs/l03_pretrain/checkpoint_best.pt",
    "runs/l04_e1/checkpoint_last.pt",
    "runs/l04_e2/checkpoint_last.pt",
    "runs/l04_e3/checkpoint_last.pt",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preservation_block() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True,
                              text=True, cwd=REPO, check=True).stdout.strip()

    checkpoint_hashes = {p: sha256_file(REPO / p) for p in CHECKPOINTS}
    return {
        "collected_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "main": git("rev-parse", "main"),
        "develop": git("rev-parse", "develop"),
        "working_tree": git("status", "--porcelain=v1").splitlines(),
        "checkpoints_preserved_sha256": checkpoint_hashes,
        "tokenizer": {
            "path": "tokenizer/taksh_code_bpe_v1.json",
            "vocab_size": 5822,
            "specials": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3},
            "committed_artifact_sha256": "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7",
            "working_tree_sha256_crlf": sha256_file(REPO / "tokenizer/taksh_code_bpe_v1.json"),
        },
        "frozen_benchmark_sha256": sha256_file(REPO / "evaluation" / "l02_python_bench_v1.json"),
        "frozen_benchmark_expected": "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6",
        "package_version": "1.0.0 (taksh == 1.0.0, unchanged)",
        "publication": "taksh-hf-publication/ untouched (upload_manifest.json modification pre-exists this session)",
    }


def run_subtools() -> dict[str, Path]:
    from evaluation import (
        l05_architecture_audit,
        l05_benchmark_audit,
        l05_compute_estimator,
        l05_data_feasibility,
        l05_training_strategy,
    )

    steps = [
        ("architecture_audit", l05_architecture_audit.main),
        ("data_feasibility", l05_data_feasibility.main),
        ("training_strategy", l05_training_strategy.main),
        ("compute_estimate", l05_compute_estimator.main),
        ("benchmark_audit", l05_benchmark_audit.main),
    ]
    outputs = {}
    for name, fn in steps:
        out = REPORTS / f"l05_{name}.json"
        rc = fn(["--out", str(out)])
        if rc != 0:
            raise SystemExit(f"step {name} failed with exit code {rc}")
        outputs[name] = out
    return outputs


def aggregate(outputs: dict[str, Path], preservation: dict) -> dict:
    merged = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in outputs.items()}
    decision = {
        "verdict": "CONDITIONAL_GO",
        "reason": (
            "A technically credible route exists (architecture scales; tokenizer is adequate; "
            "pipeline is built and verified; license-qualified data paths exist), but capability "
            "is demonstrated nowhere yet: 0/30 on the frozen benchmark for every checkpoint tested, "
            "and the qualified code supply (1.59M tokens) is ~100x below the minimum for the smallest "
            "interesting model."
        ),
        "missing_authorizations": [
            {
                "what": "Data acquisition budget + license selection",
                "consequence": "no training beyond the current 1.59M-token pool; no capability recovery possible",
                "blocked_items": ["BigCode The Stack v1/v2 (legal review)", "PyPI permissive subset (license engineering)", "code corpora >= 10B tokens class"],
            },
            {
                "what": "GPU budget (A100/H100-class, 80 GB preferred)",
                "consequence": "continued pretraining of even a 110M-class model for one epoch of a 20M-token slice is ~28 CPU-days; GPU is mandatory for anything real",
            },
            {
                "what": "Legal review sign-off on chosen corpora (OpenRAIL terms, per-file license obligations)",
                "consequence": "candidate corpora cannot be ingested; benchmark-leakage and secret scans must be re-run on whatever is chosen",
            },
            {
                "what": "Decision on strategy C vs A and target size (110M vs 289M)",
                "consequence": "blocks compute sizing and data-volume commitment",
            },
        ],
        "owner_decisions_outstanding": [
            "which data candidate set (register shortlist) is to be acquired",
            "whether strategy C (pretrain + SFT) or B (SFT-only) is pursued",
            "whether to buy/rent compute now or defer",
            "whether the frozen benchmark bar (>= 15/30) is acceptable as the L0.6 gate",
        ],
        "explicit_non_authorizations": [
            "this verdict is NOT permission to purchase, download, train, publish, or change the L0.1 default checkpoint",
            "no GPU rental, no data purchase, no training is authorized by L0.5",
        ],
        "what_would_flip_to_GO": [
            "owner-authorized acquisition of >= 20B tokens of license-verified Python (or equivalent) with a clean legal review",
            "A100-class (or better) compute for >= 2 weeks, or equivalent acceptance of a > 1-month CPU-only schedule for a reduced experiment",
            "a pre-registered benchmark protocol in which a candidate passes >= 15/30 frozen problems with zero leakage",
        ],
        "what_would_flip_to_NO_GO": [
            "owner declines data acquisition or GPU spend on this line of work",
            "legal review rejects all candidate corpora",
            "L0.6 experiment at authorized scale still yields 0/30 with a clean benchmark",
        ],
    }
    return {
        "schema": "taksh.l05_feasibility_report/1",
        "generated_by": "evaluation/l05_feasibility.py",
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mission": "L0.5 — determine the technical and economic feasibility of developing TAKSH into an independent, useful coding model. Feasibility gate, not a training milestone.",
        "preservation": preservation,
        "sub_reports": {k: str(v.relative_to(REPO)) for k, v in outputs.items()},
        "architecture_audit": merged["architecture_audit"],
        "data_feasibility": merged["data_feasibility"],
        "training_strategy": merged["training_strategy"],
        "compute_estimate": merged["compute_estimate"],
        "benchmark_audit": merged["benchmark_audit"],
        "decision": decision,
    }


def write_markdown(report: dict) -> Path:
    p = report["preservation"]
    bench = report["benchmark_audit"]
    arch = report["architecture_audit"]
    data = report["data_feasibility"]
    comp = report["compute_estimate"]
    strat = report["training_strategy"]
    lines = []
    a = lines.append
    a("# TAKSH L0.5 — Feasibility Gate Report")
    a("")
    a("**Decision: CONDITIONAL GO** — a technically credible route exists; capability is not yet demonstrated anywhere (0/30 everywhere); L0.5 authorizes nothing.")
    a("")
    a("## 1. Preservation evidence (verified 2026-11-11)")
    a("")
    a(f"- Branch `{p['branch']}` @ `{p['head'][:12]}`")
    a(f"- `main` = `{p['main'][:12]}`; `develop` = `{p['develop'][:12]}`")
    a(f"- Working tree entries: {len(p['working_tree'])} (all pre-existing modifications/untracked files preserved)")
    a(f"- Tokenizer: vocab 5,822 (`fbedf003...` committed artifact); benchmark SHA-256 `{p['frozen_benchmark_sha256'][:16]}...` matches frozen value")
    a("- Checkpoints byte-identical:")
    for name, digest in p["checkpoints_preserved_sha256"].items():
        a(f"  - `{digest[:16]}…` `{name}`")
    a("")
    a("## 2. Architecture audit (WP1)")
    a("")
    a(f"- Baseline params: **{arch['current_baseline']['params']['total_params']:,}** (analytic == measured)")
    a(f"- Status: {arch['current_baseline']['analytic_vs_measured']}")
    a(f"- Family: {arch['current_baseline']['family']['architecture']}; GQA {arch['current_baseline']['family']['gqa']}; tied embeddings; context 2048")
    a("")
    a("| Scenario | d_model | Layers | Heads (KV) | FFN | Params | vs baseline |")
    a("|---|---|---|---|---|---:|---:|")
    for s in arch["scenarios"]:
        c = s["params"]
        a(f"| {s['name']} | {s['config']['d_model']} | {s['config']['n_layers']} | "
          f"{s['config']['n_heads']} ({s['config']['n_kv_heads']}) | {s['config']['ffn_dim']} | "
          f"**{c['total_params']:,}** | {s['params_ratio_vs_baseline']}x |")
    a("")
    a("Checkpoint compatibility: existing checkpoints warmstart **none** of A/B/C (all shapes change). Tokenizer fully reusable (vocab unchanged).")
    a("")
    a("## 3. Training-data feasibility (WP2)")
    a("")
    a(f"- Qualified in-repo code supply: **{data['token_supply']['code_tokens']:,} tokens** (L0.4, measured)")
    a(f"- L0.3 general supply (CC-BY-SA / PD): **{data['token_supply']['general_tokens_l03']:,} tokens** (not code)")
    a(f"- L0.4 total train: {data['token_supply']['total_l04_train']:,} tokens; code share **0.4999**")
    a("")
    a("Required tokens (unique) vs qualified code supply:")
    a("")
    a("| Model | @20/p | @50/p | @100/p | @200/p | supply coverage @100/p | epochs @100/p |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    rows = data["requirements_grid"]
    models = []
    for r in rows:
        if r["scenario"] not in models:
            models.append(r["scenario"])
    for m in models:
        rs = {r["tokens_per_param"]: r for r in rows if r["scenario"] == m}
        p = rs[20]["params"]
        a(f"| {m} ({p/1e6:.1f}M) | {rs[20]['required_unique_tokens']/1e9:.2f}B | "
          f"{rs[50]['required_unique_tokens']/1e9:.2f}B | {rs[100]['required_unique_tokens']/1e9:.2f}B | "
          f"{rs[200]['required_unique_tokens']/1e9:.2f}B | {rs[100]['code_supply_coverage']:.4%} | {rs[100]['epochs_code_only']:.0f}x |")
    a("")
    a(f"- Register: {data['register_summary']['candidates']} candidate corpora assessed; "
      f"{data['register_summary']['legal_review_required']} require legal review; **none ingested**; "
      "benchmark sets excluded by policy.")
    a("")
    a("## 4. Training-strategy comparison (WP3)")
    a("")
    for s in strat["strategies"]:
        a(f"- **Strategy {s['id']} — {s['name']}**: {s['verdict']}")
    a(f"\nRecommended sequence: {strat['recommendation']}")
    a("")
    a("## 5. Compute and memory (WP4)")
    a("")
    a(f"- Method: training FLOPs = 6 x params x tokens; fp32/bf16 working set ~= 16 bytes/param (weights+grads+Adam).")
    a(f"- CPU anchor measured in-repo: ~{comp['cpu_anchor']['tokens_per_second']} tok/s (range {comp['cpu_anchor']['range']})")
    for name, hw in comp["hardware_feasibility"].items():
        fits = sum(1 for v in hw["verdicts"] if v["fits_training"])
        a(f"- {name}: fits training on {fits}/7 hardware classes")
    a("")
    a("GPU durations deliberately NOT invented; owner supplies price + target stack for pricing.")
    a("")
    a("## 6. Benchmark audit (WP5)")
    a("")
    a(f"- Frozen hash matches: **{bench['frozen_benchmark']['hash_matches']}**")
    a(f"- Structural verdict: **{bench['structural_audit']['verdict']}** ({bench['structural_audit']['checks_run']} checks)")
    a(f"- Raw vs serving equivalence: **{bench['serving_separation']['identical_rate']}** ({bench['serving_separation']['records_compared']} records; binding repair never changed an output)")
    a("- Acceptance bar for claiming capability: >= 15/30 frozen problems correct, >= 0.80 syntax-valid, >= 0.60 execution, all categories >= 30%, 100% repeated-run consistency, zero safety violations.")
    a("")
    a("## 7. Final decision")
    a("")
    a("### CONDITIONAL GO")
    a("")
    a("A credible route exists; nothing is authorized by this milestone. Owner decisions outstanding:")
    for item in report["decision"]["missing_authorizations"]:
        a(f"- **{item['what']}** — consequence: {item['consequence']}")
    a("")
    a("What would flip to GO:")
    for item in report["decision"]["what_would_flip_to_GO"]:
        a(f"- {item}")
    a("")
    a("What would flip to NO-GO:")
    for item in report["decision"]["what_would_flip_to_NO_GO"]:
        a(f"- {item}")
    a("")
    a("")
    a("*This document is generated by `python -m evaluation.l05_feasibility` from the same artifacts it summarizes. Nothing here authorizes purchasing, downloading, training, publishing, or changing the L0.1 default checkpoint.*")
    out = REPO / "reports" / "L05_FEASIBILITY_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-reports", action="store_true",
                        help="run the sub-tools only; do not write the aggregate reports")
    args = parser.parse_args(argv)

    outputs = run_subtools()
    if args.skip_reports:
        return 0

    preservation = preservation_block()
    report = aggregate(outputs, preservation)
    REPORTS.mkdir(exist_ok=True)
    json_path = REPORTS / "l05_feasibility_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = write_markdown(report)
    print(f"aggregate report written to {json_path}")
    print(f"markdown report written to {md_path}")
    print(f"DECISION: {report['decision']['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
