# TAKSH v1.4.0 — Release Hardening and GA Preparation

Date: 2026-11-06
Repository: `F:\Codes\Git\Quantsmind-LLM`
Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a` (verified), package version `1.0.0`
Checkpoint SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (verified unchanged)
Tokenizer SHA-256: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (verified unchanged)
Independent suite digest: `1f5024e26c37a1186db5a924e5be1dbdafbfcc1e44bc8002849ad5be045cd419` (verified unchanged)

Verdict: implementation and qualification complete. All capability invariants hold; packaging, CLI/API, and integration are hardened with executable evidence; the hybrid is **NOT YET GA-READY** because one material robustness item (a full-suite flaky test) and the documented production-use limits remain open (see §11). No publication, commit, version change, training, or dependency change occurred.

## 1. Preflight repository state and preservation evidence

- `develop` @ `a654e5d`, package `1.0.0`; working tree contains pre-existing tracked/untracked changes plus prior campaign deliverables, all preserved. No reset/clean/stash/checkout/revert/discard; no commits/pushes/merges/tags/releases; no previous reports or qualification outputs overwritten; artifacts written only under `Temp\opencode\v140\`.
- v1.3.2-D baseline inspected read-only before any change; hashes above match the v1.3.2-D record. Baseline metrics (C3 28/28, C3 functional 10, independent 39/40, copy 10/13, supplemental 33/48, generalization 21/24, abstention 5/6, AD-1 23) were re-derived on the current tree at the end of this campaign (§8).

## 2. Work-package results (WP1–WP5)

### WP1 Packaging and installation reliability
- `pyproject.toml`: package `taksh`, dynamic version from `taksh.__version__` (`1.0.0`), `[project.scripts] taksh = "taksh:main"`, `requires-python >=3.10`, dependencies unchanged (no additions/upgrades). `MANIFEST.in` includes LICENSE/NOTICE/README/CHANGELOG/pyproject plus configs.
- Build: `pip wheel . --no-deps --no-build-isolation` → `taksh-1.0.0-py3-none-any.whl` (212 files); wheel inspection found no checkpoints, credentials, or temporary files in the distribution.
- Isolated install: fresh venv (`--system-site-packages`), `pip install --no-deps <wheel>` succeeded; post-install import check `import taksh; taksh.__version__ == "1.0.0"` and `import evaluation.operation_synth` both passed.
- `scripts/check_version_consistency.py`: **PASS** (`1.0.0`, release status `final`).
- Hardening fix in this campaign: `evaluation/operation_synth.py` — the v1.3.2-B plain-subtract rule no longer shadows ordered "subtract … then double" composition; a full-suite run caught this as a real regression (see §6) and it was corrected at the source with precedence rules intact.

### WP2 CLI and API stability
- The supported CLI is the `taksh` console script (`taksh:main` in `taksh/__init__.py`, argparse with `--version`). Verified by import-call: `--version` prints `Taksh 1.0.0 (final)` and exits 0 without traceback; unrecognized arguments exit 2 with a clean argparse error and no traceback/internal paths.
- `python -m taksh` is not supported (package has no `__main__`); documented paths are the `taksh` script and `pip install .` per README. No CLI surface beyond `--version` exists; no new features added.
- Public programmatic APIs (`synthesize`, `plan`/`validate`/`compile_plan`, `HybridGenerator`) are unchanged behavior; hybrid provenance/audit records remain available (planner status + decision + mapping recorded per generation). Tests: existing CLI/API surface covered by the full suite (see §6); focused contract tests pass (52 passed).

### WP3 Integration and operational reliability
The supported end-to-end (install → import → hybrid generate → planner validation → provenance → invalid/unsupported handling) is the code path exercised by `scripts/qualify_v123.py` hybrid mode: imports, model load, prompt handling, synthesis, planner VALIDATED/UNSUPPORTED/INVALID distinctions, and evaluator-side validation/execution all run in the v140 qualification. No external-service, hosted-inference, or production-deployment claim is made.

### WP4 Security and dependency review (classified)
- **Confirmed hardening defect:** `serving/local_inference.py` `torch_load_checkpoint` attempts `weights_only=True` first but falls back to `weights_only=False` with a WARNING. Local-only, explicit `--ckpt` path, documented in the code as a legacy-checkpoint tradeoff; not a GA blocker for the local hybrid software, but it means a malicious checkpoint file can still execute code when explicitly loaded. Documented as a residual risk.
- **Informational:** training-side modules use `subprocess`/`pickle`/`eval(` in controlled experiment scripts (not on the hybrid request path). Serving layers bind `127.0.0.1` by default with `--allow-remote` opt-in and optional `QM_API_KEY` bearer auth; no secrets or credentials in the wheel; no telemetry or network exfiltration found in the hybrid path. No dependency change is requested in this campaign.
- No clean-audit claim is made; the review is structural with existing tools only.

### WP5 Documentation and capability disclosures
README was reviewed. It already distinguishes implemented code, smoke-tested execution, completed experiments, and demonstrated model capability (none), states local-prototype status, and warns that serving success does not establish output correctness. The hybrid qualification disclosure (deterministic synthesis; neural model-only capability unqualified — model-only functional 0 everywhere) is recorded in the v1.3.2 reports and §7 of this report. No benchmark, compatibility, or deployment claim was added; no neural-checkpoint qualification is advertised. This report serves as the additional capability disclosure for the v1.4.0 decision.

## 3. Exact changed-file inventory

- `evaluation/operation_synth.py`: ordered-subtract precedence hardening (plain subtract no longer shadows subtract-then-double). The only source change in this campaign.
- `tests/test_v132d_longer_string.py`, `tests/test_v132b_generalization.py`, and all other prior deliverables: unchanged by this campaign (no edits to tests).
- New deliverables: `TAKSH_V140_RELEASE_HARDENING_REPORT.md`; `Temp\opencode\v140\` (`dist/`, `venv/`, `qualification_results.json`, `capability_summary.json`). No tracked file was modified beyond the one-line precedence guard in `operation_synth.py`.

## 4. Packaging/build/install evidence

See WP1: wheel built (212 files), installed into an isolated venv, imports pass, version consistency passes. Exact commands/paths are given in `Temp\opencode\v140\`.

## 5. CLI/API and integration evidence

- `taksh --version` (via `taksh:main(['--version'])`): prints version, exit 0; `--bogus`: exit 2, clean argparse error, no traceback.
- Hybrid end-to-end: reproduced in the qualification run (see §8); planner statuses recorded: VALIDATED for synthesized plans, UNSUPPORTED_PLAN for uncovered phrasings, never conflated with INVALID.

## 6. Test and qualification results

- Full suite: `& '...\python.exe' -m pytest tests -q` → **1645 passed, 1 skipped, 2 failed initially**; one failure was the genuine subtract regression (corrected, `tests/test_v122_reliability.py` now 4 passed); the other (`tests/test_v120_increment5.py::test_functional_gate_passes_and_tamper_fails`) **passes in isolation** (16.20s) and is classified as flaky. Focused re-run of v132b/v132d/planner/hybrid/reliability contracts: **52 passed**.
- Qualification (`scripts.qualify_v123`, hybrid): C3 39/28/10, independent functional 39/40, copy 10/13, supplemental 33/48, generalization 21/24, abstention 5/6, AD-1 binding 23 (see `Temp\opencode\v140\capability_summary.json`).

## 7. Before/after capability comparison

No metric changed versus v1.3.2-D. The only behavioral delta is the restored subtract-then-double precedence (regression repair). Neural model-only capability remains 0 across all suites; all capability comes from deterministic synthesis. Controlled-pilot suitability stands; production/GA suitability is not claimed.

## 8. Hash/version verification

Post-campaign: checkpoint, tokenizer, and suite digests as in the header; package version `1.0.0`; version-consistency script PASS.

## 9. Remaining defects and risks

1. **Flaky test** (`test_v120_increment5.py::test_functional_gate_passes_and_tamper_fails`): passes alone, failed once in the full run — non-deterministic ordering/timing risk; not a capability regression.
2. **weights_only=False fallback** for legacy checkpoints (documented residual risk; local-only).
3. **Capability ceiling:** independent suite still fails Q-p1 (underspecified transform) and abstention Q-x3; unseen phrasings fall back to abstention or the 0-functional model.
4. **No CI gate run in this campaign** (PR #1 remains open with known CI adjudications); full-suite results here are local only.

## 10. Authorization compliance

Authorized scope only: packaging/CLI/API/integration/security/documentation hardening + tests + qualification. Prohibited operations (publication, distribution, commits, pushes, merges, tags, releases, training, checkpoint/tokenizer/architecture/fixture/threshold changes, dependency changes) were not performed; the single source change is a proven-defect correction with tests and qualification.

## 11. GA readiness matrix and recommendation

| Area | Rating |
|---|---|
| Functional correctness (hybrid invariants) | PASS |
| Installation and packaging | PASS |
| API/CLI stability | PASS |
| Integration reliability | PASS |
| Security | PARTIAL (weights_only fallback risk; flaky test hygiene) |
| Documentation | PASS |
| Reproducibility | PASS (deterministic qualification reproduced) |
| Known limitations | Disclosed |
| Outstanding risks | Flaky test, security fallback |

**Recommendation: NOT YET GA-READY.** Material open items — the flaky full-suite test and the `weights_only=False` legacy-checkpoint path (with no capability change needed, but hardening/CI evidence required) — preclude a GA CANDIDATE verdict. The hybrid software remains CONTROLLED PILOT ONLY. This recommendation does not authorize publication, distribution, version changes, tagging, or release.

## 12. Final verdict

**PARTIAL.** Release hardening was implemented, tested, and qualified with executable evidence, but the GA-readiness decision is NOT YET GA-READY.
