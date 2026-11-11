# TAKSH L0.7.0 — Data Acquisition and Qualification Readiness Report

**Date:** 2026-11-11
**Branch:** `feature/l060-foundation-model-qualification`
**Starting HEAD:** `9721aa62e1f15ee572658992a79149ec1f20e50b` (L0.6.2)
**Milestone type:** acquisition-readiness qualification — **NOT an acquisition, NOT a training authorization, NOT a capability claim.**
**Package version:** remains `1.0.0` — not bumped, not a release.

---

## 1. Verdict

**L0.7.0 QUALIFIED (engineering). Zero datasets were acquired, downloaded, or ingested.**
The readiness infrastructure is implemented, deterministic, and fail-closed; the only
qualified corpus in TAKSH remains the internal Python corpus (1,590,703 licensed tokens).
Every external candidate is **not acquisition-ready** (legal review required / rejected /
blocked). Acquisition execution is **permanently disabled** in this codebase
(`ACQUISITION_EXECUTION_ENABLED = False` in `evaluation/l070_acquisition_gate.py` and the
`acquisition_execution_disabled: true` invariant in `configs/data/l070_acquisition_workflow.yaml`).

This milestone establishes the *path* toward L0.7.1/L0.7.2 corpora. It is not a corpus.

---

## 2. Sources assessed (with evidence gathered 2026-11-11)

Only **metadata/license pages** were read — zero dataset bytes were accessed or downloaded.
Exactly what was retrieved and when is recorded per decision record. Anything not verifiable
without accepting upstream gates or downloading content is marked unresolved.

| Source | Evidence read | Result |
|---|---|---|
| **BigCode The Stack v1** (`bigcode/the-stack`) | HF dataset card (2026-11-11): gated; ToU pass-through of original licenses; 193 permissive SPDX licenses (copyleft removed v1.1); 358 languages; ~6TB near-dedup; opt-out ablation duty | `LICENSE_REVIEW`; `HUMAN_LEGAL_REVIEW_REQUIRED`; commercial use **UNKNOWN** |
| **BigCode The Stack v1 (alt)** (`bigcode/bigcode-stack-v1`) | HTTP 401 — gated/removed; terms unverifiable without acceptance | registered under The Stack v1 record; noted as inaccessible reference |
| **BigCode The Stack v2** (`bigcode/the-stack-v2`) | HF dataset card (2026-11-11): gated; SWHID-only metadata (content behind Software Heritage S3; bulk requires **Software Heritage + INRIA agreement**); SWH principles for LLM-for-code training (2023-10-19 statement) apply; permissive per-file detection (Blue Oak + ScanCode); 658 languages; 67.53TB raw / 32.1TB dedup / ~900B tokens (reported); opt-out = the removal mechanism; `v2.2.0` (2026-07-29) | `PROVENANCE_REVIEW`; `HUMAN_LEGAL_REVIEW_REQUIRED`; commercial use **UNKNOWN** |
| **CodeSearchNet** (`github/CodeSearchNet`) | GitHub repo (2026-11-11): tooling/docs **MIT**; corpus = Go/Java/JavaScript/PHP/Python/Ruby from public non-fork **redistribution-licensed** GitHub repos (~6M functions; paper arXiv:1909.09436); per-file licenses NOT embedded in corpus rows; CodeXGLUE datasets carry a research-oriented **C-UDA** caveat | `METADATA_REVIEW`; `HUMAN_LEGAL_REVIEW_REQUIRED`; commercial use **UNKNOWN** |
| **PyPI permissive aggregate** | PyPI is a repository, not a dataset; ToU grants no corpus rights; no bulk-ingestion path exists without per-package adjudication | `DEFERRED`; `REJECTED` for training-use purposes |
| **Karkain corpus** | `F:/Codes/Git/Karkain` (owner-internal, read-only, e612ede); SPEC/compiler pinned 1.1.0; `kcc.exe` → `0xC0000135`; all compile-pipeline commands exit 3 | `BLOCKED`; zero corpus; not eligible |
| **TAKSH internal l04 corpus** | `data/l04_code_corpus/PROVENANCE.md` (measured L0.4): 1,590,703 Python tokens; per-slice licenses verified; 4 benchmark-colliding slices quarantined | `ACQUIRED` (prior milestone); `APPROVED_WITH_CONDITIONS` (internal research use) |

**No candidate is recommended for acquisition.** Comparison table: `reports/l070_candidate_comparison.json`.

---

## 3. License qualification approach

- Five+ axes assessed **separately** per source (source license, file/package licenses, dependency licenses, redistribution, commercial use, derivatives, ML-training permission, ToS, dataset-specific restrictions, copyright/attribution). They are never collapsed into one boolean (validator enforces structure).
- Decisions: `APPROVED | APPROVED_WITH_CONDITIONS | HUMAN_LEGAL_REVIEW_REQUIRED | REJECTED | UNKNOWN`. Unknown fails closed.
- **Overall consistency rule:** `APPROVED*` is invalid if any axis is fail-closed (`HUMAN_LEGAL_REVIEW_REQUIRED`/`UNKNOWN`/`REJECTED`) — this caught and corrected one internal inconsistency during qualification (internal corpus `dependency_licenses`, now recorded as explicitly out-of-scope with note).
- Records are engineering decision records, **explicitly not legal advice**; licensed counsel review is a precondition for any acquisition, named in the records.
- Existing heuristic licensing gate (`evaluation/l060_training_readiness.py`) remains the shared license-name oracle; L0.7.0 adds the decision-record layer above it.
- **Unresolved legal questions** (verbatim in records): express commercial-training grant for The Stack v1/v2; INRIA-agreement scope for commercial use of an SWH mirror; opt-out freshness SLA vs static training snapshots; liability for misdetected per-file licenses/PII; CodeSearchNet corpus-level license (MIT tooling ≠ corpus rights; C-UDA applicability); per-file license reconstruction strategy for CodeSearchNet rows.

## 4. Language coverage and gaps (53 + Karkain)

Full machine-readable matrix: `reports/l070_coverage_matrix.json` (validated from committed `l051_languages.yaml`; byte-identical registry).

- **Qualified corpus: 1 language in 1 domain** — Python (1,590,703 tokens; 384 samples; internal, licensed; acquisition state `internal_verified`).
- **52 of 53 languages: zero qualified tokens.** Karkain (54th): zero.
- **L1.0 (initial 16):** all 16 have unverified candidate coverage (Stack v2 conservative subset / CodeSearchNet 6-language overlap / PyPI Python-only); none verified; 15 of 16 have no in-repo tokens at all.
  - python (acquired internal) · javascript, typescript, c, cpp, java, csharp, go, rust, php, ruby, swift, kotlin, dart, sql, bash (candidate-covered, unverified, legal review open)
- **Enterprise/legacy (vbnet, vba, cobol, delphi, plsql, tsql …):** candidate-covered by Stack only (unverified); legal review open.
- **Scientific/functional (r, matlab, julia, haskell, fsharp, ocaml, clojure, lisp …):** candidate-covered by Stack (unverified).
- **Systems/embedded/games/blockchain (zig, ada, solidity, gdscript, …):** partially candidate-covered (conservative subset), mostly thin.
- **Emerging/specialist (mojo, micropython, gleam, odin, vlang, hack, pony, crystal …):** intentionally **not** assigned Stack coverage — the full language list was not fetched (it is in a side CSV; fetching it is allowed metadata but was unnecessary); listed as candidate-unverified or NO_CANDIDATE per language.
- Accounting is kept separate: qualified vs candidate vs unverified vs approved vs actually-acquired vs training-eligible = **1,590,703 + 0 + 0 + 0 + 1,590,703 + 1,590,703**. Candidates never counted.
- Supporting technologies (20) remain separate from languages; zero qualified tokens; never counted as the 53/54.

## 5. Data-quality qualification framework (pre-acquisition)

`evaluation/l070_data_quality.py` — deterministic heuristics that emit **flags for human review**, never "clean" certifications:

- UTF-8/BOM/null-byte decode; empty/corrupt detection; alphanumeric fraction; extreme line length (minification); generated-code markers (`@generated`, `DO NOT EDIT`, protoc/swagger/grpc); license-marker presence (signal only); Python syntax validity via `ast` where toolchain exists (other languages: explicitly SKIPPED, not guessed); exact duplicate detection (normalized SHA-256); near-duplicate detection (token-shingle Jaccard ≥ 0.8 configurable); secret/credential patterns (AWS `AKIA…`, `github_pat_`, `ghp_`, Slack `xox…`, private-key blocks, `password=`/`api_key=` literals) — **flagged, never auto-cleared; human review mandatory**;
- separation of **automated** vs **human review** in every report; runs on synthetic fixtures (labeled) and, later, on licensed slices (as L0.4 did manually).

## 6. Benchmark contamination protection

- Pinned registry: `configs/data/l070_contamination_registry.yaml` — frozen Python bench (`854c7ca8…`, re-hashed at run time and in CI of this milestone), multilingual 64-task bench (`7a839060…`), KARK-BENCH schema (`fe6acbcd…`);
- normalized fingerprinting (UTF-8 → lowercase → comment-strip → whitespace-collapse → SHA-256); task-level fingerprints of both frozen benchmarks captured in memory at audit time;
- synthetic contaminated fixture (embedding a real frozen-bench problem) is **detected: exact=1, near=1 → BLOCKED**; clean fixture → `CLEAN (fixtures only)`;
- any future candidate corpus must pass this screen; any hit blocks training eligibility until resolved; exclusion lists are append-only.

## 7. Corpus planning (honest, deterministic)

`configs/data/l070_corpus_plan.yaml` + `evaluation/l070_corpus_planner.py`:

| Strategy | Achievable total | Unplannable (available=0) | Impossible (target > available, available>0) |
|---|---|---|---|
| equal_language | 200,000 (Python capped) | 53/54 | 52 |
| proportional_to_availability | 1,590,703 | 53/54 | 0 |
| quality_weighted | 1,590,703 | 53/54 | 0 |
| hybrid_language_family | 200,000 (Python capped) | 53/54 | 52 |

- Targets are parameters, not existence claims. No fabricated totals anywhere; Karkain target hard-wired to 0; exposure ceiling 4× (L0.6.1 consistent); no winning strategy claimed; verdict: **no actionable strategy until acquisition is authorized and measured**.

## 8. DLCAE integration (steps 3–8 of the 10-step flow)

`evaluation/l070_dlcae_sources.py` — qualification jobs per (language, source): gates =
{source covers language, provenance VERIFIED, license review complete, comprehension complete, contamination screen, owner authorized}.
Results today: python←l04 = 3/6 gates (eligible=**false** — contamination screen + owner gate outstanding); python←stack-v2 = 0/6; java←stack-v2 = 0/6; karkain←karkain-corpus = 1/6 (covers-language only).
**Zero sources eligible**, zero manifests emittable, registry statuses never auto-upgraded; `NullKnowledgeSource` discipline preserved; external retrieval stays optional and is not a RAG substitute.

## 9. Karkain data strategy

`configs/data/l070_karkain_data_strategy.yaml` — identity pinned (.kark; SPEC **1.1.0**; compiler **1.1.0 windows/amd64 Stable**; KIR v1 SSA; `TAKSH-KARK-BENCH` 0 tasks; status DISCOVERED); 9 corpus requirement groups (stdlib examples, positive/negative programs, borrow/move cases, codegen, debug/repair, repo-level, KIR-annotated, diagnostics) — **all NOT_INSTANTIATED; zero counts anywhere**; compiler-backed validation mandatory; contamination exclusions pinned to KARK-BENCH; blockers recorded (`0xC0000135`; exit-3 pipeline; owner authorization absent); synthetic fixtures labeled synthetic (3 tiny .kark-shaped strings, schema-tests only). No Karkain repository changes made, ever.

## 10. Acquisition authorization workflow

Stages: DISCOVERY → TECHNICAL → LEGAL → PROVENANCE → **OWNER_APPROVAL_REQUIRED** → QUALIFIED_FOR_ACQUISITION → ACQUISITION_AUTHORIZED → EXECUTION → POST-VERIFICATION.
- 16-field request dossier; no auto-advance past owner stage (validator enforces);
- `QUALIFIED_FOR_ACQUISITION` is explicitly **not** an authorization;
- even a fully-filled dossier with a synthetic `owner_authorization_id` returns **execution disabled** — the invariant must be flipped by a future milestone + explicit out-of-band owner decision;
- zero authorizations exist in the registry (`ACQUIRED` only for grandfathered internal l04 corpus).

## 11. Resource and storage readiness

`evaluation/l070_resource_readiness.py` (planning; nothing provisioned/downloaded): host = 3.91 GB RAM (4.2 GB total), **170.4 GiB free disk**; Stack v1 (~1.43TB page footprint / ~6TB dedup) and Stack v2 (67.5TB raw / 427GB metadata-only repo) are **not stageable here**; example estimate for 1.4TB candidate: ~2.1TB workspace (labeled estimate); recommended 10-stage preprocessing pipeline documented; reservations: **paid storage NONE, cloud resources NONE, network downloads NONE** (verified by report invariants and test assertions).

## 12. Focused tests and results

```
python -m pytest tests/test_l070_data_readiness.py -q      → 50 passed (21 required areas)
python -m pytest tests/test_l0601_pipeline.py tests/test_l060_foundation.py \
  tests/test_l060r1_architecture_alignment.py tests/test_m6_resources.py \
  tests/test_l062_inference.py tests/test_l070_data_readiness.py -q
                                                          → 229 passed
python -m evaluation.l070_observability                     → 12/12 tools exit 0
```
(Areas covered: registry schema, snapshot identity, license-record validation, fail-closed
missing-license behavior, training-permission validation, commercial/redistribution axis
separation, owner-approval enforcement, acquisition-authorization enforcement, provenance
completeness, multilingual coverage math, language/technology separation, candidate-vs-qualified
accounting, exact+near duplicate detection, benchmark contamination rejection, data-quality
validation, corpus-plan determinism, DLCAE handoff, Karkain contract pinning, L0.6.1 manifest
compatibility, resource reporting, audit-trail/report generation.)

## 13. Preservation evidence (before and after)

| Artifact | Method | Hash |
|---|---|---|
| Frozen Python benchmark | SHA-256 (working tree) | `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6` — unchanged |
| Multilingual benchmark (64 tasks) | SHA-256 | `7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e` — unchanged |
| `v180_sft` checkpoint | streaming SHA-256 | `9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666` — unchanged |
| Tokenizer committed blob (LF) | `git cat-file blob HEAD:tokenizer/taksh_code_bpe_v1.json` | `fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7` — unchanged |
| `l060_foundation.yaml` | SHA-256 | `01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3` — unchanged |
| 53-language registry | SHA-256 | `075d4b568f052c8ebede69ccaf21cc4d0fd15a4f01df455f1431505c354ef39d` — unchanged |
| KARK-BENCH schema | SHA-256 | `fe6acbcd9a41d5e113595d4fdb675cc616342da30029401fe16602d6c157d481` — unchanged |
| taksh-hf-publication/ | only pre-existing CRLF modification (`upload_manifest.json`) | preserved, not staged |
| develop / main | untouched | ✓ |

## 14. Exact new files (all L0.7.0-owned)

`configs/data/l070_source_registry.yaml`, `l070_license_decisions.yaml`, `l070_acquisition_workflow.yaml`, `l070_corpus_plan.yaml`, `l070_contamination_registry.yaml`, `l070_karkain_data_strategy.yaml` ·
`evaluation/l070_{source_registry,license_qualification,provenance,coverage_matrix,data_quality,contamination,corpus_planner,dlcae_sources,acquisition_gate,karkain_strategy,resource_readiness,candidate_comparison,observability}.py` ·
`tests/test_l070_data_readiness.py` ·
`reports/l070_{source_registry,license_decisions,provenance_assessment,coverage_matrix,data_quality,contamination_audit,corpus_plan,dlcae_sources,acquisition_workflow,karkain_strategy,resource_readiness,candidate_comparison,observability}.json` ·
`docs/L070_DATA_ACQUISITION_READINESS_REPORT.md` (this file)

No pre-existing file modified. Web access in this milestone was limited to metadata/license
pages listed in §2 (2026-11-11). Nothing else fetched; no downloads of any size; no scraping.

## 15. Explicit attestation

**No datasets were acquired, downloaded, mirrored or ingested. No training or fine-tuning occurred.
No GPU was used or provisioned. No paid API or storage was used. No publication, tag, release,
merge to `develop`/`main`, or Karkain-repository modification occurred. The frozen benchmark was
not executed (0/30 ceiling stands). Package version remains `1.0.0`.**

## 16. Blockers and owner decisions required

1. **Legal counsel review** — none of the external candidates can advance without counsel sign-off
   (The Stack v1/v2 terms, SWH/INRIA agreement scope, CodeSearchNet corpus license, PyPI aggregate).
2. **Owner acquisition decision** — whether any external corpus is ever authorized for TAKSH; if so:
   which scope (language subset), which service path (SWH+S3 vs HF gated pull), and at what cost.
3. **Infrastructure for acquisition** — SWH/INRIA agreement + (if authorized) credentials, large
   staging disk (~1–2TB), filtered-slice strategy; this host cannot stage them.
4. **Karkain toolchain** — kcc repair (or different host) before any Karkain corpus work;
   KARK-BENCH bar still undefined; owner authorization for ingestion absent.
5. **Python breadth** — the only verified corpus is 1.59M Python tokens; owner must decide whether
   internal expansion (e.g., more permissive slices from L0.4-style curation) precedes any external ask.
6. **Canonical foundation-111m** — still untrained; L0.7.x corpus work is a precondition of L0.7.1/L0.7.2.

L0.7.0 stops here. L0.7.1 is not started.
