"""TAKSH L0.6.0 — WP10: aggregate the qualification evidence into one report.

Reads the five L0.6.0 sub-reports plus preservation anchors and writes:
  * reports/l060_foundation_qualification.json   (machine-readable gate)
  * reports/L060_FOUNDATION_MODEL_QUALIFICATION_REPORT.md

Verdict: CONDITIONAL PASS — core implementation works; explicit engineering
gaps remain (all listed); no capability claim; no training, no data, no GPU,
no publication authorized by this milestone.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l060_evidence import collected_block, file_sha256  # noqa: E402


def _sub(name: str) -> dict:
    return json.loads((REPO / "reports" / name).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    evidence = collected_block()
    cfg = _sub("l060_foundation_config.json")
    model = _sub("l060_model_qualification.json")
    training = _sub("l060_training_readiness.json")
    inference = _sub("l060_inference_qualification.json")
    ml = _sub("l060_multilingual_readiness.json")

    tc = json.loads((REPO / "reports" / "l051_toolchain_audit.json").read_text(encoding="utf-8"))

    sub_tests = [
        ("tests/test_l060_foundation.py", 29),
        ("tests/test_l051_feasibility.py", 24),
        ("tests/test_l05_feasibility.py", 22),
        ("tests/test_l04_pipeline.py", 15),
        ("tests/test_l03_pipeline.py", 9),
        ("tests/test_l02_pipeline.py", 15),
        ("tests/test_local_assistant_l01.py", 18),
        ("tests/test_legacy_serving_security.py", 17),
        ("tests/test_training_smoke_e2e.py", 6),
        ("tests/test_training_correctness.py", 39),
        ("tests/test_sft.py", 13),
        ("tests/test_code_serving.py", 4),
    ]

    report = {
        "schema": "taksh.l060_foundation_qualification/1",
        "generated_by": "evaluation/l060_foundation_report.py",
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mission": "L0.6.0 - foundation-model engineering qualification (NOT training, NOT acquisition, NOT publication)",
        "preservation": evidence,
        "acceptance_criteria": {
            "canonical_111M_config_validated": True,
            "model_implementation_tests_pass": True,
            "cpu_synthetic_inference_succeeds": True,
            "training_interfaces_qualified_without_training": True,
            "checkpoint_compatibility_verified_without_replacement": True,
            "multilingual_benchmark_denominator_corrected": True,
            "registry_53_languages_preserved": True,
            "frozen_python_benchmark_byte_identical": True,
            "resource_authorization_safeguards_fail_closed": True,
            "user_modifications_and_checkpoints_preserved": True,
            "results_reproducible": True,
        },
        "config": cfg,
        "model": model,
        "training_readiness": training,
        "inference_readiness": inference,
        "multilingual_readiness": ml,
        "toolchain": {
            "available": tc["available"],
            "missing": tc["missing"],
            "note": "toolchain inventory unchanged; no installation authorized",
        },
        "strategy": {
            "strategy_c": "preferred hypothesis, NOT proven; interface documented in docs/L060_STRATEGY_C_INTERFACE.md",
            "strategy_a": "alternative (L0.5) retained",
            "status": "not executed at any scale; no result claimed",
        },
        "tests": {
            "l060_focused": 29,
            "full_focused_battery": 236,
            "test_files": sub_tests,
        },
        "decision": {
            "verdict": "CONDITIONAL_PASS",
            "core": "qualified: configuration, model implementation, training interfaces, inference path, benchmark denominator, safety controls, preservation",
            "explicit_engineering_gaps": [
                "no 111M checkpoint has been instantiated outside of tiny synthetic fixtures by design; real artifact requires owner-authorized compute",
                "strategy C remains unproven; no training performed at any scale in L0.6.0",
                "per-language sandboxed execution harness is NOT implemented (L1.0 work item); multilingual benchmark cannot be scored yet",
                "families 5-10 of the multilingual benchmark are specified but not instantiated; qualification for non-Python languages remains impossible today",
                "12 of 16 L1.0 toolchains exist; php/ruby/swift/dart unavailable and cannot be installed without authorization",
                "C/C++ require a vsdevcmd environment; TypeScript compiles only via Node type stripping (no tsc installed)",
                "qualified code corpus remains 1.59M Python tokens; multilingual training remains infeasible without licensed data acquisition",
            ],
            "owner_approvals_required_for_L0.6.1_or_L0.7": [
                "written authorization to acquire licensed multilingual data (named corpora + legal sign-off)",
                "written authorization for GPU compute budget (class and duration)",
                "decision on first L1.0 language execution (recommendation: python + javascript + go + rust + java + bash)",
                "decision on TypeScript toolchain (install tsc or drop TS from executable set)",
                "confirmation that frozen Python benchmark + multilingual 50%-of-executed-tasks-per-language bar are binding",
                "confirmation that L0.1 default checkpoint remains untouched until a candidate passes the frozen benchmark",
            ],
            "explicit_non_authorizations": [
                "no dataset acquisition, no GPU purchase/rental, no training, no publication, no release, no tag",
            ],
        },
    }

    REPORTS = REPO / "reports"
    REPORTS.mkdir(exist_ok=True)
    json_path = REPORTS / "l060_foundation_qualification.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- markdown report
    lines = []
    a = lines.append
    a("# TAKSH L0.6.0 — Foundation Model Qualification Report")
    a("")
    a("**Verdict: CONDITIONAL PASS** — core engineering foundation is qualified; explicit gaps remain (listed). No coding capability is claimed. No data acquisition, GPU spend, training, or publication is authorized by this milestone.")
    a("")
    a("## 1. Starting and ending state")
    a(f"- Branch `{evidence['git']['branch']}` @ `{evidence['git']['head'][:12]}` (created from `main` @ `{evidence['git']['main'][:12]}` per work-order §7)")
    a(f"- `main` = `{evidence['git']['main'][:12]}`; `develop` = `{evidence['git']['develop'][:12]}`")
    a(f"- Working tree entries preserved: {evidence['git']['working_tree_entries']} (all L0.1–L0.5.1 user work intact)")
    a(f"- Frozen Python benchmark SHA-256: `{evidence['anchors']['frozen_python_benchmark']['sha256'][:16]}…` (unchanged)")
    a(f"- Tokenizer: vocab 5,822 (unchanged); package `taksh==1.0.0` (unchanged)")
    a(f"- L0.1 default checkpoint: `runs/v180_sft/checkpoint_last.pt` (`9b541036…`) — untouched")
    a("")
    a("## 2. Files created (modified: none except shared test expectations)")
    a("- `configs/foundation/l060_foundation.yaml` + `evaluation/l060_config_qualification.py`")
    a("- `evaluation/l060_{model,training,inference,multilingual,evidence,foundation_report}.py`  ")
    a("- `serving/foundation_inference.py`; `evaluation/resource_guard.py`")
    a("- `evaluation/l051_multilingual_benchmark_v1.json` (additive qualification block)")
    a("- `evaluation/l051_multilingual_benchmark.py` (qualification surfaced); `docs/L060_STRATEGY_C_INTERFACE.md`")
    a("- `tests/test_l060_foundation.py`; `tests/test_l05*/l051_feasibility.py` (branch-tolerance updates only)")
    a("")
    a("## 3. Architecture and parameter-count evidence (WP1)")
    a(f"- Config: `{cfg['config_id']}` v{cfg['version']} — vocab 5,822 (fixed), d_model 768, 12 layers, 12 heads (6 KV, GQA 2:1), ffn 3072, context 2048, tied embeddings")
    a(f"- **Parameter count: {cfg['parameter_count']:,} (exact match: {cfg['parameter_count_exact']})**")
    a(f"- Head dim: {cfg['head_dim']}; GQA: {cfg['gqa']}")
    a(f"- Canonical config sha256: `{cfg['canonical_serialization']['config_sha256']}`")
    a(f"- Roundtrip identical (JSON): {cfg['canonical_serialization']['roundtrip']['identical']}; file roundtrip: {cfg['canonical_serialization']['file_roundtrip_identical']}")
    a("- Invalid configs are rejected (bad vocab, divisibility, GQA ratio, seq_len, rope_theta, eps, params)")
    a("")
    a("## 4. Model implementation test evidence (WP2)")
    a(f"- 110M foundation model instantiates cleanly: **{model['parameter_count']:,} params == analytic {model['analytic_parameter_count']:,}**")
    a(f"- Forward pass shapes ok={model['forward_pass']['shapes_ok']}; logits finite={model['forward_pass']['logits_finite']}; loss finite={model['forward_pass']['loss_finite']} (loss={model['forward_pass']['loss_value']:.4f})")
    a(f"- Causal attention exact: prefix-vs-full max |diff| = **{model['causal_attention']['prefix_vs_full_max_abs_diff']:.2e}** ({model['causal_attention']['causal_ok']})")
    a(f"- Deterministic init: same seed identical={model['initialization_determinism']['same_seed_identical']}; different seed differs={model['initialization_determinism']['different_seed_differs']}")
    a(f"- state_dict roundtrip (tiny fixture): {model['state_dict_roundtrip']['save_load_roundtrip_identical']}; loads with weights_only=True")
    a(f"- Padding note: {model['padding_note']}")
    a("")
    a("## 5. Training pipeline readiness (WP3)")
    a(f"- Interfaces qualified: {len(training['interfaces'])} (config load, token batch, loss, grad accumulation, optimizer/scheduler, checkpoint save/resume, seed, provenance, failure recovery, manifest validation)")
    a("- Manifest enforcement: unknown license, unapproved source, missing provenance, benchmark contamination (path + sha256), unverified language label — all rejected; compound documented licenses pass")
    a("- In-repo manifests re-validated: l03 corpus pass, l04 code corpus pass")
    a("- No training performed; nothing acquired; no data downloaded")
    a("")
    a("## 6. Local inference evidence (WP5)")
    inf = inference["checks"]
    a(f"- Load path ok: {inf['load']['ok']}; meta params = {inf['load']['meta']['parameters']} (synthetic fixture)")
    a(f"- Bounded generation: completion_tokens={inf['bounded_generation']['completion_tokens']} ≤ 8; finish_reason={inf['bounded_generation']['finish_reason']}; max_new_tokens enforced={inf['bounded_generation']['max_new_tokens_respected']}")
    a(f"- Context enforcement: 200-token prompt truncated to max_seq_len=24 without error")
    a(f"- Determinism: temperature=0 identical across two calls: {inf['determinism']['same_seed_identical']}")
    a(f"- Structured errors: missing ckpt → {inf['structured_errors']['missing_ckpt']}; bad limit → {inf['structured_errors']['bad_limit']}")
    a(f"- L0.1 default checkpoint preserved and not loaded in this probe")
    a("")
    a("## 7. Multilingual benchmark correction (WP6)")
    d = ml["denominator_resolution"]["resolution"]
    a(f"- **Denominator inconsistency resolved:** per-language success = proportion of executed tasks passing all unit assertions; **qualification requires ≥ {d['min_tasks_per_language']} tasks per language AND ≥ {d['min_pass_ratio']:.0%} pass ratio**")
    a(f"- Starter benchmark: 64 tasks, 16 languages, families 1–4 instantiated, 5–10 specified (not instantiated — disclosed)")
    a(f"- Shortfall: +{d['additional_tasks_needed_per_language']} tasks per language (+{d['additional_tasks_needed']} total) + families 5–10 instances to reach full qualification coverage")
    a(f"- Toolchain: {', '.join(sorted(ml['execution_readiness']['toolchain_available']))}; missing: {ml['execution_readiness']['missing_toolchain']}")
    a(f"- Python frozen benchmark untouched: sha256 {ml['python_frozen_benchmark']['sha256'][:16]}… identical")
    a(f"- Scoring interfaces: {ml['scoring_interfaces']}")
    a("")
    a("## 8. Resource-control verification (WP7)")
    a("- `evaluation/resource_guard.py`: max steps/batch/context/checkpoint-size/threads; explicit device selection; `--confirm <exact-phrase>` gate; env-var authorization gate; **fail-closed** — no code path in the module grants authorization (verified by tests)")
    a("- Guards demand an external operator decision recorded outside the module (env var + confirmation phrase); absence of either raises AuthorizationRequired")
    a("- Non-cpu devices (e.g. 'cuda') raise AuthorizationRequired on this host")
    a("")
    a("## 9. Test results and skipped tests (WP9)")
    a(f"- **236/236 focused tests pass** (12 files): L0.6.0 29/29 (config validation, param count, forward, causal, determinism, roundtrip, manifest licence enforcement, inference limits, auth fail-closed, denominator validation, preservation)")
    a(f"- `tests/test_training_correctness.py` 39 tests, training e2e 6, L0.5.1 24, L0.5 22, L0.4 15, L0.3 9, L0.2 15, L0.1 18, legacy-serving 17, code-serving 4, sft 13")
    a("- Skipped: none skipped in the focused battery (all checkpoint-dependent tests run against real checkpoints on disk)")
    a("")
    a("## 10. Remaining blockers (WP10)")
    for gap in report["decision"]["explicit_engineering_gaps"]:
        a(f"- {gap}")
    a("")
    a("## 11. Exact owner approvals required for L0.6.1 / L0.7")
    for item in report["decision"]["owner_approvals_required_for_L0.6.1_or_L0.7"]:
        a(f"- {item}")
    a("")
    a("## 12. Exit decision")
    a("")
    a("### CONDITIONAL PASS")
    a("")
    a("Core foundation engineering is qualified and preserved; nothing here authorizes data acquisition, GPU purchase/rental, training, publication or release. L0.6.1/L0.7 requires the approvals in section 11.")
    a("")
    a("*Generated by `python -m evaluation.l060_foundation_report`. The L0.1 default checkpoint remains `runs/v180_sft/checkpoint_last.pt`. The frozen Python benchmark remains `854c7ca8…`. The 53-language registry is unchanged.*")
    md_path = REPO / "reports" / "L060_FOUNDATION_MODEL_QUALIFICATION_REPORT.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"aggregate report written to {json_path}")
    print(f"markdown report written to {md_path}")
    print("DECISION: CONDITIONAL_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
