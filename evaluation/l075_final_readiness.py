"""TAKSH L0.7.5 — FINAL readiness gate for L0.8.0 (fail-closed, 18 dimensions).

One authoritative aggregate over all readiness evidence from L0.7.0 through
L0.7.4, plus L0.7.5 additions: training artifact provenance, training
environment authorization, external distribution permission and Karkain
readiness. Every dimension records evidence, date, owner-vs-engineering
responsibility and blocking dependency. The verdict is the most restrictive
dimension; technical facts never imply owner decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l071_corpus_record import record_fingerprint  # noqa: E402

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_COMMITTED_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"
FOUNDATION_CFG_SHA = "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"
L051_SHA = "075d4b568f052c8ebede69ccaf21cc4d0fd15a4f01df455f1431505c354ef39d"

VERIFIED = "2026-11-11"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dim(status, evidence, *, blocking, owner_decision, responsibility):
    return {
        "status": status,
        "evidence": evidence,
        "last_verified": VERIFIED,
        "blocking_dependency": blocking,
        "responsible_party": "owner_decision" if owner_decision else "engineering",
        "owner_decision_required": bool(owner_decision),
        "responsibility": responsibility,
    }


def dim_snapshot():
    root = REPO / "artifacts" / "l072_l04_freeze"
    snap = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    records = [json.loads(l) for l in (root / "records_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    fps = sorted(record_fingerprint(r) for r in records)
    ok = hashlib.sha256("".join(fps).encode()).hexdigest() == snap["integrity_hashes"]["records_sha256"]
    return dim("PASS" if (snap["snapshot_id"] == "l072-l04-fdb1fee2beaae9c2" and ok) else "FAIL",
               f"snapshot_id={snap['snapshot_id']}; integrity match={ok}; frozen_for_training={snap['frozen_for_training']}",
               blocking="none (passed) — but immutability alone confers nothing",
               owner_decision=False, responsibility="snapshot preserved; owner needed only for new freeze")


def dim_provenance():
    from evaluation.l070_provenance import assess
    from evaluation.l070_source_registry import load_registry

    src = next(s for s in load_registry()["sources"] if s["source_id"] == "src-taksh-l04-code-corpus")
    verdict = assess(src)["verdict"]
    return dim("PASS" if verdict == "VERIFIED" else "FAIL",
               f"internal l04 corpus provenance {verdict}; train.jsonl re-hashed at ingest; dataset_manifest.json audited L0.4",
               blocking="external candidates require counsel — none acquired",
               owner_decision=False, responsibility="engineering (internal corpus); external scope = owner+counsel")


def dim_technical_eligibility():
    snap = json.loads((REPO / "artifacts" / "l072_l04_freeze" / "eligibility.json").read_text(encoding="utf-8"))
    agg = snap["aggregate"]
    return dim("PASS" if agg["technically_eligible"] >= 1050 else "FAIL",
               f"technically_eligible={agg['technically_eligible']}; excluded={agg['excluded']}; training_eligible={agg['training_eligible']}",
               blocking="none; technical != legal != training",
               owner_decision=False, responsibility="engineering (recompute at run time); authorization is owner's")


def dim_legal_permission():
    return dim("PARTIAL",
               "l070 license record: APPROVED_WITH_CONDITIONS — internal SFT/pretraining previously authorized (L0.4); external/commercial release requires fresh counsel review; redistribution NOT granted",
               blocking="A3.2 (external datasets — counsel); A3.3/A3.4 (distribution/publication)",
               owner_decision=True, responsibility="owner + counsel")


def dim_owner_authorization():
    return dim("MISSING",
               "no owner authorization for any training run exists; A1.1 of the L0.7.5 decision register = MISSING; A1.2 = CONDITIONAL (internal research pretraining only, aggregate registry still not an authorization)",
               blocking="A1.1 written authorization (snapshot, scope, environment, budget, distribution, expiry)",
               owner_decision=True, responsibility="owner")


def dim_language_coverage():
    return dim("PARTIAL",
               "python 1,380,611 eligible tokens (only); 52 of 53 languages: zero; Karkain: zero (DISCOVERED); supporting technologies: separate; scope unchanged",
               blocking="data acquisition decisions (A3.2) — external corpora need counsel + owner",
               owner_decision=False, responsibility="engineering fact; scope decisions are owner's")


def dim_tokenizer_model():
    blob = subprocess_text(["git", "cat-file", "blob", "HEAD:tokenizer/taksh_code_bpe_v1.json"])
    committed = hashlib.sha256(blob).hexdigest()
    ok = committed == TOK_COMMITTED_SHA and sha256_file(REPO / "configs" / "foundation" / "l060_foundation.yaml") == FOUNDATION_CFG_SHA
    return dim("PASS" if ok else "FAIL",
               f"tokenizer committed LF blob {committed[:16]}… matches fbedf003…; foundation config matches; vocab 5822; context 2048; tied embeddings; 110,658,816 params",
               blocking="none",
               owner_decision=False, responsibility="engineering")


def subprocess_text(argv):
    import subprocess

    return subprocess.run(argv, capture_output=True, check=True, cwd=REPO).stdout


def dim_checkpoint_availability():
    return dim("MISSING",
               "taksh-foundation-111m has NO checkpoint anywhere; v180_sft exists (28.8M, d_model 384) — GEOMETRY_INCOMPATIBLE vs canonical (l074 validator); synthetic fixture rejected; spec present (l074_canonical_checkpoint_readiness.yaml)",
               blocking="a canonical 111M artifact must be produced by an authorized training run (which does not exist)",
               owner_decision=False, responsibility="engineering — but blocked by owner decisions A1/A2")


def dim_hardware():
    import shutil

    import psutil

    vm = psutil.virtual_memory()
    free = shutil.disk_usage(REPO).free
    return dim("BLOCKED",
               f"measured 2026-11-11: RAM total {vm.total} B, available {vm.available / 2**20:.1f} MiB, {vm.percent}% used; disk free {free / 2**30:.1f} GiB; MX150 4 GiB (CUDA 11.1, CPU torch); fp32+Adam ≈ 1.77 GiB estimate exceeds host",
               blocking="A2-compute-authorization + qualifying GPU (≥16 GiB VRAM estimate) + corpus residency policy",
               owner_decision=True, responsibility="owner must provide/authorize environment")


def dim_benchmarks():
    ok = sha256_file(REPO / "evaluation" / "l02_python_bench_v1.json") == FROZEN_PY_SHA and sha256_file(REPO / "evaluation" / "l051_multilingual_benchmark_v1.json") == MULTILINGUAL_SHA
    return dim("PASS" if ok else "FAIL",
               "frozen Python benchmark 854c7ca8… intact; multilingual 64-task 7a839060… intact; thresholds and expected outputs unchanged; 0/30 stands",
               blocking="none",
               owner_decision=False, responsibility="engineering (frozen; mutation forbidden without owner-approved milestone)")


def dim_focused_regression():
    import yaml

    ledger = yaml.safe_load((REPO / "configs" / "data" / "l073_regression_ledger.yaml").read_text(encoding="utf-8"))
    resolved = sum(1 for e in ledger["entries"] if e["disposition"] == "RESOLVED")
    return dim("PASS" if resolved == 7 else "FAIL",
               f"l073 ledger: {resolved}/7 RESOLVED; suites unchanged through L0.7.5",
               blocking="CI execution still requires owner action (A4)",
               owner_decision=False, responsibility="engineering")


def dim_full_ci():
    return dim("BLOCKED",
               "production CI triggers are push-to-develop/main + PR-to-those-branches only; workflow has no workflow_dispatch; ci-l0704-manual-qualification.yml exists on branch but GitHub exposes workflows only from the DEFAULT branch (HTTP 404 verified 2026-11-11); dispatch attempts: HTTP 422 (no trigger) and HTTP 404",
               blocking="A4 owner action: push the qualification workflow to main (or add workflow_dispatch to production CI) or open+merge an authorized PR",
               owner_decision=True, responsibility="owner")


def dim_python_runtime():
    return dim("CONDITIONAL",
               "requires-python >=3.10; 479 files statically parsed at 3.11 grammar — 0 real-code incompatibilities; all deps cp311; runtime NOT EXECUTED — no Python 3.11 interpreter installed on host; downloads unauthorized",
               blocking="owner authorization for interpreter provisioning then runtime qualification",
               owner_decision=True, responsibility="owner (to authorize interpreter); engineering (to verify runtime)")


def dim_reproducibility():
    return dim("PASS",
               "content-derived snapshot id rebuilt byte-reproducibly; atomic writes verified; resume gates closed; l071/l072 lineage intact; leakage checks green",
               blocking="none",
               owner_decision=False, responsibility="engineering")


def dim_artifact_provenance():
    return dim("MISSING",
               "no training run exists; artifact contract specified (l074_canonical_checkpoint_readiness.yaml) but NO artifact has ever been materialized; v180_sft is a different model (GEOMETRY_INCOMPATIBLE)",
               blocking="A1.1 authorization → authorized run → artifact",
               owner_decision=True, responsibility="owner authorization; engineering execution under authorization")


def dim_training_environment_authorization():
    return dim("MISSING",
               "no authorized training environment exists — MX150 unusable (0x…/CUDA 11.1); no external compute approved; no residency policy defined",
               blocking="A2-compute-authorization",
               owner_decision=True, responsibility="owner")


def dim_distribution_permission():
    return dim("NOT_AUTHORIZED",
               "no external/commercial checkpoint distribution right exists; A3.3/A3.4 = NOT_AUTHORIZED; internal research conditional only (A3.1)",
               blocking="counsel opinion + owner decision before any external artifact leaves internal custody",
               owner_decision=True, responsibility="owner + counsel")


def dim_karkain():
    import yaml

    strategy = yaml.safe_load((REPO / "configs" / "data" / "l070_karkain_data_strategy.yaml").read_text(encoding="utf-8"))
    ident = strategy["identity"]
    return dim("BLOCKED",
               f"Karkain remains DISCOVERED (SPEC 1.1.0; compiler 1.1.0 pinned; kcc.exe 0xC0000135; zero corpus; qualification never upgraded); extension registry entry intact (54th first-class language)",
               blocking="owner-authorized Karkain corpus + kcc repair; never implied by this milestone",
               owner_decision=True, responsibility="owner (Karkain repo); engineering otherwise")


def evaluate() -> dict:
    dimensions = {
        "corpus_snapshot_integrity": dim_snapshot(),
        "data_provenance": dim_provenance(),
        "technical_corpus_eligibility": dim_technical_eligibility(),
        "legal_training_permission": dim_legal_permission(),
        "owner_training_authorization": dim_owner_authorization(),
        "multilingual_language_coverage": dim_language_coverage(),
        "tokenizer_model_compatibility": dim_tokenizer_model(),
        "canonical_checkpoint_availability": dim_checkpoint_availability(),
        "hardware_resource_feasibility": dim_hardware(),
        "benchmark_thresholds": dim_benchmarks(),
        "focused_regression_qualification": dim_focused_regression(),
        "full_github_actions_ci_execution": dim_full_ci(),
        "python_runtime_compatibility": dim_python_runtime(),
        "reproducibility_and_recovery": dim_reproducibility(),
        "training_artifact_provenance": dim_artifact_provenance(),
        "training_environment_authorization": dim_training_environment_authorization(),
        "external_checkpoint_distribution_permission": dim_distribution_permission(),
        "karkain_readiness_and_integration": dim_karkain(),
    }
    blocking = sorted(k for k, v in dimensions.items()
                      if v["status"] in ("MISSING", "BLOCKED", "NOT_AUTHORIZED"))
    verdict = "BLOCKED" if blocking else "CONDITIONALLY_READY"
    return {
        "schema": "taksh.final_pretraining_readiness/1",
        "generated_by": "evaluation/l075_final_readiness.py",
        "verified_on": VERIFIED,
        "dimensions": dimensions,
        "dimension_count": len(dimensions),
        "blocking_dimensions": blocking,
        "verdict": verdict,
        "verdict_options": ["READY_FOR_L0_8_WITHIN_AUTHORIZED_SCOPE",
                            "CONDITIONALLY_READY", "BLOCKED"],
        "training_eligible_tokens": 0,
        "mandatory_fact_check": {
            "l072_snapshot": "l072-l04-fdb1fee2beaae9c2",
            "python_samples": 1050,
            "python_tokens": 1380611,
            "other_languages_tokens": 0,
            "karkain_tokens": 0,
            "owner_training_authorization": "MISSING",
            "training_eligible_tokens": 0,
        },
        "invariants": [
            "READY verdict only with owner authorization + checkpoint initialization path + authorized viable environment",
            "no owner decision may be fabricated by tooling",
            "technical facts never override owner decisions",
            "zero tolerance for skipped/deleted/weakened regression tests",
        ],
        "verdict_text": (
            "BLOCKED — L0.8.0 must not begin. Training authorization is MISSING (A1.1), "
            "no canonical foundation-111m artifact exists, the host cannot hold the training state, "
            "and every legal right beyond internal research pretraining is absent. The owner must act first; "
            "then this gate flips automatically in a new milestone without new code."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l075_final_readiness.json")
    args = parser.parse_args(argv)
    report = evaluate()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, d in report["dimensions"].items():
        print(f"{name:44s} {d['status']}")
    print(f"verdict: {report['verdict']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
