# TAKSH — Recovery Defect Register (2026-11-11)

**Milestone:** CI/CD & Documentation Recovery. **Baseline:** `d2f0d1443ddb959429a2232a8d240457c2f90590`.
**Scope rules:** nothing here authorizes training/release/merge/publishing; Historical Failure Set 001 is tracked as seven individually traceable defects. No defect is silently dropped.
**Statuses:** RESOLVED · BLOCKED · DEFERRED · OPEN. **Fix commit:** this recovery (`recovery-2026-11-11`; final SHA in `reports/taksh_cicd_documentation_recovery.json` and the audit report header).

---

## A. Historical Failure Set 001 (owner-supplied, 7/7 verified)

Original failing context: `CI` → `unit-and-smoke` on `develop` (runs `42144960532` etc., 2026-10-21/2026-11-06), job exceeded the 30-minute budget after `Unit tests` failed (7 failed, 1295 passed, 39 skipped).

| ID | Defect | Affected files | Evidence | Root cause (verified) | Resolution | Regression test | Status |
|---|---|---|---|---|---|---|---|
| R-H1 | `tests/test_m5_adapters_syntax.py::test_external_adapter_does_not_download_by_default` — adapter preflight ordered weights-loading before license/dependency check | `serving/providers.py`, adapters | owner-supplied failure: `assert 'nothing was fetched' in ...'outdated transformers'` | preflight order: dependency first → RAM → license; transformers imported before weights-discovery could proceed offline | reordered: license → dependency-free weights discovery → RAM → transformers (L0.7.3 fix, present at HEAD) | `tests/test_m5_adapters_syntax.py::test_external_adapter_does_not_download_by_default` — **passes** (recovery run) | RESOLVED |
| R-H2 | `tests/test_m6_sovereign.py::test_available_ram_is_the_binding_constraint` — host-specific RAM assumption broke determinism | `tests/test_m6_sovereign.py`, sovereign RAM accounting | owner-supplied failure: `assert False is False` generator edge | non-deterministic local measurement asserted directly | RAM measurement made injectable; test asserts on injected fixture, not host | `tests/test_m6_sovereign.py::test_available_ram_is_the_binding_constraint` — **passes** (recovery run) | RESOLVED |
| R-H3 | `tests/test_v120_python_data.py::test_g7_manifest_has_no_timestamp_or_absolute_path` — absolute path leaked into manifest | `data/python_v1/manifest.py` | owner-supplied failure: `/tmp/pytest-of-runner/pytest-0/python_v10/bench.jsonl` found in output | manifest recorded the caller's output root | manifest builder is content-addressed and path-free by construction; deterministic builders (`build_manifest` docstring: *no timestamps, no absolute paths*) | `tests/test_v120_python_data.py::test_g7_manifest_has_no_timestamp_or_absolute_path` — **passes** | RESOLVED |
| R-H4 | `tests/test_v150_coding_foundation.py::test_geometry_inspector_reports_a_matching_triple` — checkpoint absent | `tests/fixture_s2_checkpoint.py`, `tests/test_v150_coding_foundation.py` | missing file `runs/m7_pretrain/checkpoint_last.pt` | CI lacks the gitignored private artifact; test required it implicitly | synthetic fixture at `tests/fixture_s2_checkpoint.py` (`code_m3_9m` geometry, step 300, deterministic seed 20261111) | `tests/test_v150_coding_foundation.py::test_geometry_inspector_reports_a_matching_triple` — **passes** | RESOLVED |
| R-H5 | `tests/test_v150_coding_foundation.py::test_tokenizer_artifact_and_model_geometry_are_unchanged` — artifact missing ⇒ embedding mismatch | same as R-H4 | `embedding_rows None` | same missing private artifact | same fixture; byte-level geometry verified via fixture metadata | `tests/test_v150_coding_foundation.py::test_tokenizer_artifact_and_model_geometry_are_unchanged` — **passes** | RESOLVED |
| R-H6 | `tests/test_v160_authorization_and_baseline.py::test_g10_geometry_inspector_accepts_the_s2_triple` — same | same as R-H4/R-H5 | same | same | same | `tests/test_v160_authorization_and_baseline.py::test_g10_geometry_inspector_accepts_the_s2_triple` — **passes** | RESOLVED |
| R-H7 | `tests/test_v180_corpus_qualification.py::test_sampled_decoding_is_reproducible_with_a_seed` — FileNotFoundError (checkpoint) | `tests/test_v180_corpus_qualification.py`, seeded decode path | missing checkpoint | same as above; seeded reproducibility requires a resident geometry | same fixture; seeded sampling asserted deterministic | `tests/test_v180_corpus_qualification.py::test_sampled_decoding_is_reproducible_with_a_seed` — **passes** | RESOLVED |

**Additional invariants pinned by recovery:** the synthetic checkpoint fixture is clearly labeled `code_m3_9m` and `must never be substituted for` the canonical 111M artifact (its own header). `v180_sft` was never touched, read for hashing or used as a stand-in. `tests/test_l100eng3_end_to_end.py::test_canonical_checkpoint_mismatch_via_v180` (geometry mismatch) passes — the canonical checkpoint gate remains enforced.

## B. Recovery-scope defects found and fixed

| ID | Category | Sev | Affected files | Evidence | Root cause | Resolution | Regression test | Status |
|---|---|---|---|---|---|---|---|---|
| R-CI-01 | CI/CD | High | `.github/workflows/ci.yml` | §4 of audit; `gh workflow list --all` | Missing top-level `permissions:` (default grants broader token permissions than needed; least-privilege violated) | `permissions: contents: read` + ownership/visibility comments | `tests/test_ci_workflows.py::test_ci_workflow_parses_and_is_least_priv_read_only` | RESOLVED |
| R-CI-02 | CI/CD | High | `docs/audit/TAKSH-CICD-DOCUMENTATION-RECOVERY.md` §4 | GitHub Actions: `workflow_dispatch` only lists workflows on the default branch; `ci-l0704-manual-qualification.yml` exists only on the feature branch → remote CI impossible from feature branch | GitHub policy, not a repo bug; visibility only via default branch | documented exact owner actions; no agent-side dispatch attempted (forbidden) | `tests/test_ci_workflows.py::test_manual_workflow_dispatch_input_defaults_to_feature_branch` + audit §4 | BLOCKED (owner) |
| R-CI-03 | CI/CD | Medium | `.github/workflows/ci-l0704-manual-qualification.yml` | stale chain only covered L0.6.0→L0.7.4 | milestones advanced; manual path left behind | focused chain extended L0.6.0→L1.0.0-ENG3 (incl. ENG1/2/3 + recovery tests); concurrency pinned to non-cancellation; dispatch-visibility guidance added | `tests/test_ci_workflows.py::test_manual_workflow_qualified_chain_covers_all_recent_milestones` | RESOLVED |
| R-CI-04 | CD/Release | Medium | `docs/STATUS.md`, `README.md` | both banners showed only M7-era status; new ENG3 state invisible | stale documentation after ENG1–3 landed | additive 2026-11-11 TAKSH status banners inserted; historical content preserved verbatim | presence checks in this recovery (grep); no generated tests (docs) | RESOLVED |
| R-CI-05 | Documentation | Low | `docs/audit/` | no audit deliverables existed | absent | created `TAKSH-CICD-DOCUMENTATION-RECOVERY.md`, this register, machine JSON; Sphinx rebuild verified (98 pages, exit 0); no deploy | `sphinx-build` exit 0 (evidence) | RESOLVED |
| R-CI-06 | Verification | Low | workflow YAML syntax | no parser in repo exercised workflows | untested workflow engineering | YAML parse + structural assertions added | `tests/test_ci_workflows.py` (`_load`) | RESOLVED |

## C. Additional historical defects discovered

| ID | Category | Sev | Evidence | Status |
|---|---|---|---|---|
| R-H08 | CI runtime budget | Medium | `develop` runs 2026-10-21/2026-11-06: job `unit-and-smoke` exceeded 30m after `Unit tests` failed (annotations show both HF001 failure and job timeout) | OPEN — structural: full-suite 30m budget vs this host (~4.2 GB RAM) and free-tier runner limits; manual path already uses 45m and focused chain; full-suite runtime to be re-measured remotely once dispatch is unblocked (owner action, R-CI-02) |
| R-H09 | CI infra | Low | Node 20 forced to Node 24; `ubuntu-latest` migrating to Ubuntu 26 (Oct 2026) | OPEN — upstream migration annotations; no agent-side action without default-branch edits; documented |
| R-H10 | Artifact hygiene | Low | `runs/smoke/*` artifacts reported missing in run annotation (produced only after unit pass) | LOW/OPEN — expected consequence of R-H08; no code defect |

## D. Documentation defects evaluated (no code change required)

| ID | Item | Disposition |
|---|---|---|
| R-DOC-01 | `README.md` stale M7-era status (2026-10-20) | RESOLVED — additive banner (R-CI-04) |
| R-DOC-02 | `docs/STATUS.md` same | RESOLVED — additive banner (R-CI-04) |
| R-DOC-03 | Sphinx `docs/source` covers only M2→M7; ENG milestones not embedded | DEFERRED — owner content decision; build verified, nothing deployed |
| R-DOC-04 | `docs/ROADMAP.md` vs roadmap registry authority question | ACCEPTED BY DESIGN (roadmap changelog Rev 1.0); no contradiction of release state |
| R-DOC-05 | Python 3.11 runtime never executed on this host (workflow requests 3.11) | OPEN — documentation will state 3.11 unverified locally; no install attempted (authorized reason) |
| R-DOC-06 | `pyproject.toml` `requires-python` hygiene unverified | OPEN — release-tooling concern; no metadata change without owner decision |

## E. Owner actions required (exact)

| # | Action | Why |
|---|---|---|
| 1 | Add `feature/l060-foundation-model-qualification` to `push:` trigger of `ci.yml` (on default branch) OR dispatch the manual-qualification workflow from `main`/`develop` against this branch | GitHub exposes `workflow_dispatch` only for default-branch workflows (evidence in audit §4) |
| 2 | Merge the Historical Failure Set 001 fix (already on feature branch) into `develop` after review | Otherwise every future CI run re-fails and times out (R-H08) |
| 3 | Decide whether to enable required checks/status on the feature branch | Protection settings outside agent authority |
| 4 | Decide whether to expand `docs/source` to the L0.6.x→ENG3 corpus | Documentation publishing decision; deploy is owner-only |
| 5 | Decide the Python 3.11 vs host-3.14 support statement for `pyproject.toml` | Release metadata is owner-owned |
| 6 | Authorize the canonical 111M training (A1.1-C reopen / A2) when ready | Blocks any coding-capability measurement; otherwise MODEL_UNAVAILABLE stands |

## F. Summary counts

| Metric | Count |
|---|---|
| Historical defects (Failure Set 001) | **7 — all VERIFIED RESOLVED at HEAD** |
| Recovery defects fixed | 6 (R-CI-01…R-CI-06) |
| Historical defects open/blocked | 3 (R-H08, R-H09, R-H10) — all LOW/MEDIUM; none weaken tests |
| Documentation defects evaluated | 6 (2 RESOLVED, 1 DEFERRED, 1 BY-DESIGN, 2 OPEN) |
| Critical severity | none |
| Production truth | GA NOT established; capability NOT_EVALUATED; Karkain DISCOVERED; 0/30 benchmark; develop/main untouched |
