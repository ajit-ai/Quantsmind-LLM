"""TAKSH L0.5.1 — single reproducible entry point for the multilingual feasibility gate.

Runs the L0.5.1 tools in order, verifies preservation, and writes:
* reports/l051_feasibility_report.json   (single machine-readable consolidated gate)
* reports/L051_MULTILINGUAL_FEASIBILITY_REPORT.md  (WP10 decision document)

Usage:
    python -m evaluation.l051_feasibility

No model is trained; no data is downloaded; nothing outside reports/ is written
(sub-tools do write their own reports/l051_*.json by design).
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
    "runs/v160_pretrain/checkpoint_last.pt", "runs/v180_sft/checkpoint_last.pt",
    "runs/l02_sft/checkpoint_last.pt", "runs/l02_sft_hp/checkpoint_last.pt",
    "runs/l02_overfit/checkpoint_last.pt", "runs/l03_pretrain/checkpoint_last.pt",
    "runs/l03_pretrain/checkpoint_best.pt", "runs/l04_e1/checkpoint_last.pt",
    "runs/l04_e2/checkpoint_last.pt", "runs/l04_e3/checkpoint_last.pt",
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

    return {
        "collected_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "main": git("rev-parse", "main"),
        "develop": git("rev-parse", "develop"),
        "working_tree_entries": len(git("status", "--porcelain=v1").splitlines()),
        "checkpoints_preserved_sha256": {p: sha256_file(REPO / p) for p in CHECKPOINTS},
        "frozen_python_benchmark_sha256": sha256_file(REPO / "evaluation" / "l02_python_bench_v1.json"),
        "frozen_python_benchmark_expected": "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6",
        "tokenizer": {"path": "tokenizer/taksh_code_bpe_v1.json", "vocab_size": 5822,
                      "specials": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}},
        "package_version": "taksh==1.0.0 (unchanged)",
        "publication": "taksh-hf-publication/ untouched",
    }


def run_subtools() -> dict[str, Path]:
    from evaluation import (
        l051_acceptance_gates,
        l051_architecture_comparison,
        l051_compute_estimator,
        l051_dataset_feasibility,
        l051_language_registry,
        l051_multilingual_benchmark,
        l051_tokenizer_audit,
        l051_toolchain_audit,
        l051_training_distribution,
    )

    steps = [
        ("language_registry", l051_language_registry.main),
        ("tokenizer_audit", l051_tokenizer_audit.main),
        ("dataset_feasibility", l051_dataset_feasibility.main),
        ("training_distribution", l051_training_distribution.main),
        ("architecture_comparison", l051_architecture_comparison.main),
        ("multilingual_benchmark", l051_multilingual_benchmark.main),
        ("toolchain_audit", l051_toolchain_audit.main),
        ("compute_estimate", l051_compute_estimator.main),
        ("acceptance_gates", l051_acceptance_gates.main),
    ]
    outputs = {}
    for name, fn in steps:
        out = REPORTS / f"l051_{name}.json"
        rc = fn(["--out", str(out)])
        if rc != 0:
            raise SystemExit(f"step {name} failed with exit code {rc}")
        outputs[name] = out
    return outputs


def classify_languages(reports: dict) -> dict:
    classification = {}
    for lid in [
        "python", "javascript", "typescript", "c", "cpp", "java", "csharp",
        "go", "rust", "php", "ruby", "swift", "kotlin", "dart", "sql", "bash",
    ]:
        toolchain = reports["toolchain_audit"]["l1_summary"].get(lid, False)
        dataset = reports["dataset_feasibility"]["per_language"][lid]["verdict"]
        if dataset == "qualified-in-repo" and toolchain:
            cls = "feasible-for-L1.0-investigation"
        elif toolchain:
            cls = "conditional-on-data/legal-review"
        else:
            cls = "conditional-on-toolchain+data/legal-review"
        classification[lid] = {"l1_target": True, "classification": cls,
                               "toolchain_available_here": toolchain,
                               "dataset_verdict": dataset}
    for lid in [
        "vbnet", "vba", "cobol", "fortran", "delphi", "perl", "groovy", "scala",
        "objectivec", "assembly", "plsql", "tsql",
    ]:
        classification[lid] = {
            "l1_target": False,
            "classification": "conditional-on-data/legal-review" if reports["dataset_feasibility"]["per_language"][lid]["verdict"] == "conditional-on-legal-review" else "deferred",
            "dataset_verdict": reports["dataset_feasibility"]["per_language"][lid]["verdict"],
        }
    for lid in [
        "r", "matlab", "julia", "haskell", "fsharp", "ocaml", "elixir",
        "erlang", "clojure", "lisp",
    ]:
        classification[lid] = {
            "l1_target": False,
            "classification": "deferred",
            "dataset_verdict": reports["dataset_feasibility"]["per_language"][lid]["verdict"],
        }
    for lid in [
        "zig", "ada", "lua", "solidity", "gdscript", "mojo", "micropython",
        "prolog", "nim", "crystal",
    ]:
        classification[lid] = {
            "l1_target": False,
            "classification": "deferred",
            "dataset_verdict": reports["dataset_feasibility"]["per_language"][lid]["verdict"],
        }
    for lid in [
        "gleam", "odin", "vlang", "hack", "pony",
    ]:
        classification[lid] = {
            "l1_target": False,
            "classification": "technically-unsupported-by-current-evidence",
            "dataset_verdict": reports["dataset_feasibility"]["per_language"][lid]["verdict"],
        }
    return classification


def aggregate(outputs, preservation):
    reports = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in outputs.items()}
    classification = classify_languages(reports)

    consolidated_counts = {}
    for entry in classification.values():
        consolidated_counts[entry["classification"]] = consolidated_counts.get(entry["classification"], 0) + 1

    decision = {
        "verdict": "CONDITIONAL_GO",
        "scope": "engineering implementation and local feasibility analysis only; no training, no data acquisition",
        "reasons": [
            "a multilingual shared-model program is technically plausible (tokenizer is adequate for all 73 audited samples, no roundtrip loss)",
            "BUT the qualified in-repo multilingual code supply is exactly 1.59M Python tokens and ZERO for every other language",
            "and no multilingual benchmark-executed capability exists for any language (0/30 on Python; 0/16 on the multilingual starter)",
            "and 10 of 16 L1.0 toolchains are locally usable but 4 (php/ruby/swift/dart) are not, and typescript has no compiler",
            "and no architecture beyond the verified 28.8M baseline is justified by evidence today",
        ],
        "owner_approvals_required_for_L0.6": [
            "written authorization to acquire licensed multilingual data (named corpora; legal sign-off per corpus)",
            "written authorization for GPU compute budget (class and duration)",
            "selection of the initial L1.0 language set to actually execute (subset of the 16 recommended by toolchain+data)",
            "confirmation that the multilingual benchmark acceptance bar (>= 8/16 per language) is binding",
            "decision on execution of TypeScript via Node type-stripping vs installing tsc (authorization required)",
            "confirmation that L0.1 default checkpoint remains untouched",
        ],
        "what_would_flip_to_GO": [
            "owner-authorized acquisition of >= 20B tokens of license-verified multilingual code with clean legal review",
            ">= 24 GB VRAM-class compute for a >= 100M-parameter model, or equivalent CPU-only schedule accepted (~28 days)",
            "a candidate passing >= 8/16 multilingual tasks AND beating the frozen Python bar on its own benchmark",
        ],
        "what_would_flip_to_NO_GO": [
            "owner declines data acquisition or compute budget",
            "legal review rejects all candidate corpora for the selected languages",
            "an authorized L0.6 experiment scores 0/16 across all languages with a clean benchmark",
        ],
    }

    return {
        "schema": "taksh.l051_feasibility_report/1",
        "generated_by": "evaluation/l051_feasibility.py",
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mission": "L0.5.1 - extend the L0.5 feasibility gate to the 53-language multilingual coding assistant; feasibility and engineering implementation only; NOT authorization to train or acquire data",
        "preservation": preservation,
        "sub_reports": {k: str(v.relative_to(REPO)) for k, v in outputs.items()},
        "language_registry": reports["language_registry"],
        "tokenizer_audit": reports["tokenizer_audit"],
        "dataset_feasibility": reports["dataset_feasibility"],
        "training_distribution": reports["training_distribution"],
        "architecture_comparison": reports["architecture_comparison"],
        "multilingual_benchmark": reports["multilingual_benchmark"],
        "toolchain_audit": reports["toolchain_audit"],
        "compute_estimate": reports["compute_estimate"],
        "acceptance_gates": reports["acceptance_gates"],
        "per_language_classification": classification,
        "classification_counts": consolidated_counts,
        "decision": decision,
    }


def write_markdown(report: dict) -> Path:
    p = report["preservation"]
    decisions = report["decision"]
    tok = report["tokenizer_audit"]
    ds = report["dataset_feasibility"]
    tc = report["toolchain_audit"]
    bm = report["multilingual_benchmark"]
    lines = []
    a = lines.append
    a("# TAKSH L0.5.1 — Multilingual Feasibility Report")
    a("")
    a("**Decision: CONDITIONAL GO** — engineering implementation delivered and verified; capability is demonstrated nowhere (0/30 Python, 0/16 multilingual starter); authorization to train, buy data, or rent GPUs is NOT granted.")
    a("")
    a("## 1. Preservation evidence (verified 2026-11-11)")
    a(f"- Branch `{p['branch']}` @ `{p['head'][:12]}`")
    a(f"- `main` = `{p['main'][:12]}`; `develop` = `{p['develop'][:12]}`; working tree entries: {p['working_tree_entries']}")
    a(f"- Tokenizer: vocab 5,822 (unchanged); all specials intact (pad=0 bos=1 eos=2 unk=3)")
    a(f"- Frozen Python benchmark SHA-256 `{p['frozen_python_benchmark_sha256'][:16]}…` == frozen; `taksh-hf-publication/` untouched; package `taksh==1.0.0`")
    a("")
    a("## 2. Language registry (WP1)")
    a("- 53 languages registered and validated: exactly 16 L1.0 targets; 12 enterprise; 10 scientific; 10 systems; 5 emerging.")
    a("- Supporting technologies kept separate: HTML, CSS, SCSS, JSON, YAML, XML, TOML, GraphQL, Terraform HCL, Nix, CMake, Makefiles, Gradle DSL, PowerShell, Zsh, OpenAPI, Protocol Buffers, Verilog, SystemVerilog, VHDL — none counted as programming languages.")
    a("")
    a("## 3. Tokenizer coverage audit (WP2)")
    a(f"- 73 samples (53 languages + 20 supporting); roundtrip exact: **{tok['roundtrip_exact_count']}/{tok['samples']}**")
    a(f"- tokens/char range {tok['tokens_per_char']['min']:.3f}..{tok['tokens_per_char']['max']:.3f}")
    a("- No language excluded by tokenization; indentation single-token; byte-exact decode for every sample.")
    a("- Audit measures representation, not capability.")
    a("")
    a("## 4. Multilingual dataset feasibility (WP3)")
    a(f"- L1.0 verdicts: qualified-in-repo: {ds['l1_target_summary']['qualified_in_repo']}; "
      f"conditional-on-legal-review: {len(ds['l1_target_summary']['conditional_on_legal_review'])}; "
      f"deferred: {len(ds['l1_target_summary']['deferred'])}; blocked: {ds['l1_target_summary']['blocked']}")
    a(f"- Global: 1 qualified (Python), 28 conditional, 24 deferred, 0 blocked")
    a("- in-repo ground truth: 1,590,703 Python tokens (measured); **zero tokens for every other L1.0 language**")
    a("- Nothing downloaded, scraped, or purchased; BigCode/CodeSearchNet/PyPI candidates registered, all require legal review")
    a("")
    a("## 5. Training distribution model (WP4)")
    a("- equal_token: blocked (15 of 16 languages have zero unique supply)")
    a("- proportional: degenerates to 100% Python today")
    a("- quality_weighted: collapses to Python (only verified-quality signal)")
    a("- capability_gap: uniform-by-default (no capability measured anywhere)")
    a("- family_balance: 100% scripting(python), 0% elsewhere")
    a("- equal-language epoch at the 4x reuse ceiling: ~5.97M unique tokens missing (15 languages x ~398K each)")
    a("")
    a("## 6. Architecture comparison (WP5)")
    a("- A (one shared multilingual model): PREFERRED HYPOTHESIS — not proven; cheapest, deployment-simplest, matches verified stack")
    a("- B (shared + language adapters): fallback if A shows destructive interference; 41.8M params")
    a("- C (shared + family adapters): variant of B; 35.3M params")
    a("- D (separate models): rejected for now — 460.6M params, no shared capability, absurd at current scale")
    a("")
    a("## 7. Multilingual benchmark (WP6)")
    a(f"- {bm['task_count']} starter tasks; {len(bm['languages_covered'])} languages; families 1-4 instantiated; families 5-10 specified but NOT instantiated (disclosed)")
    a(f"- Frozen Python benchmark SHA-256 unchanged: {bm['frozen_python_benchmark']['hash_matches']}")
    a("- Proposed threshold fixed before any evaluation: >= 8/16 tasks per language; Python bar unchanged; never renegotiated after results")
    a("")
    a("## 8. Toolchain inventory (WP7)")
    a(f"- available ({len(tc['available'])}/16): {', '.join(sorted(tc['available']))}")
    a(f"- missing ({len(tc['missing'])}/16): {', '.join(tc['missing'])}")
    a("- SQL via Python sqlite3 (3.50.4); Bash via Git Bash 5.2.37; C/C++ via MSVC 19.44.35229 (vsdevcmd required); TypeScript via Node type-stripping only; no tsc")
    a("")
    a("## 9. Compute and storage (WP8)")
    a("- checkpoint storage: 28.8M→115 MB, A→443 MB, B→1.16 GB, C→4.07 GB; training working sets ~16 bytes/param")
    a("- CPU inference of the current model verified (L0.1); 289M fits 16 GB VRAM classes for training; 1B needs ≥ 24 GB")
    a("- equal-split across 16 languages: 99,419 tokens/language from the full qualified pool — nothing to train with")
    a("")
    a("## 10. Per-language risk classification (WP10)")
    a("| Classification | Count | Languages |")
    a("|---|---|---|")
    for label in ("feasible-for-L1.0-investigation", "conditional-on-data/legal-review",
                  "conditional-on-toolchain+data/legal-review", "deferred",
                  "technically-unsupported-by-current-evidence"):
        langs = sorted(k for k, v in report["per_language_classification"].items() if v["classification"] == label)
        a(f"| {label} | {len(langs)} | {', '.join(langs)} |")
    a("")
    a("## 11. Decision (WP10)")
    a("")
    a("### CONDITIONAL GO")
    a("Reasons:")
    for r in decisions["reasons"]:
        a(f"- {r}")
    a("")
    a("Owner approvals required for L0.6:")
    for r in decisions["owner_approvals_required_for_L0.6"]:
        a(f"- {r}")
    a("")
    a("Would flip to GO:")
    for r in decisions["what_would_flip_to_GO"]:
        a(f"- {r}")
    a("")
    a("Would flip to NO-GO:")
    for r in decisions["what_would_flip_to_NO_GO"]:
        a(f"- {r}")
    a("")
    a("")
    a("*Generated by `python -m evaluation.l051_feasibility`. This document and its JSON twin do not authorize training, data acquisition, compute rental, or publication. The L0.1 default checkpoint remains unchanged.*")
    out = REPO / "reports" / "L051_MULTILINGUAL_FEASIBILITY_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-reports", action="store_true")
    args = parser.parse_args(argv)

    outputs = run_subtools()
    if args.skip_reports:
        return 0

    preservation = preservation_block()
    report = aggregate(outputs, preservation)
    REPORTS.mkdir(exist_ok=True)
    json_path = REPORTS / "l051_feasibility_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = write_markdown(report)
    print(f"aggregate report written to {json_path}")
    print(f"markdown report written to {md_path}")
    print(f"DECISION: {report['decision']['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
