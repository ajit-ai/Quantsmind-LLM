# TAKSH v1.4.2 — Hugging Face PUBLIC ALPHA Publication Preparation

Date: 2026-11-07
Repository: `F:\Codes\Git\Quantsmind-LLM`
Baseline: branch `develop`, HEAD `a654e5dc13dc77e925b3981d09792bade43a970a` (verified), package version `1.0.0`
Checkpoint SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` (verified unchanged)
Tokenizer SHA-256: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` (verified unchanged)
Independent suite digest: `1f5024e26c37a1186db5a924e5be1dbdafbfcc1e44bc8002849ad5be045cd419` (verified unchanged)

Verdict: publication artifacts prepared and locally validated. **NOT YET GA-READY.** Upload-ready but awaiting explicit owner authorization before any Hugging Face repository creation, upload, or publication.

## 1. Preflight and preservation evidence

- `develop` @ `a654e5d`, package `1.0.0`; all pre-existing tracked and untracked changes preserved.
- No reset/clean/stash/checkout/revert/discard; no commits/pushes/merges/tags/releases/publication.
- No previous reports or qualification artifacts overwritten.
- No checkpoint, tokenizer, architecture, or training data modifications.
- No training or fine-tuning performed.

## 2. Inspection of publication artifacts

### Package metadata
- Package name: `taksh`
- Package version: `1.0.0` (unchanged)
- License: Apache-2.0 (`LICENSE`, `NOTICE`)
- Repository: `https://github.com/ajit-ai/Quantsmind-LLM`
- Entry points: `taksh`, `taksh-openai`

### Built distribution artifacts (in `dist/`)
| Artifact | Size | SHA-256 |
|---|---|---|
| `taksh-1.0.0-py3-none-any.whl` | 336,332 | `047ad9278dfc95cb8b6eff9b21299dd7cd7ecfe83ea117dc26e237617955d8c9` |
| `taksh-1.0.0.tar.gz` | 418,840 | `341225b9cf0f08ce5accef5f843e24009bc12d54395ce383643294c2f1d15a28` |

### Checkpoint and tokenizer provenance (NOT CLEARED for distribution)

| Asset | SHA-256 | License status | Distribution cleared? |
|---|---|---|---|
| `artifacts/taksh-s2-experimental/taksh-s2.pt` | `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` | `model_weights_license: not cleared` | **No** |
| `tokenizer/qm_code_bpe_v0.json` | `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` | `training_data_license: not cleared` | **No** |

From `artifacts/taksh-s2-experimental/artifact_manifest.json`:
- `training_data_license`: **not cleared**
- `model_weights_license`: **not cleared**
- `distribution_approved`: **false**
- `publication_status`: **not published**

From `artifacts/taksh-s2-experimental/LICENSE_STATUS.txt`:
> NOT APPROVED FOR DISTRIBUTION pending license and provenance approval.

### Conclusion on distribution
**The S2 checkpoint and tokenizer vocabulary are NOT cleared for redistribution.** Neither is included in the Hugging Face publication directory. Only source code (Apache-2.0) is published.

## 3. Hugging Face publication directory

Created at: `taksh-hf-publication/`

Structure (453 files, 3,789,853 bytes total):
```
taksh-hf-publication/
├── README.md                          # HF model card with YAML front matter
├── LICENSE                            # Apache-2.0
├── NOTICE
├── pyproject.toml                     # package metadata (dynamic version 1.0.0)
├── requirements.txt                   # runtime dependencies
├── MANIFEST.in                        # package manifest
├── pytest.ini                         # test configuration
├── conftest.py                        # test configuration
├── .gitattributes                     # HF export attributes (excludes weights/tokenizer/data)
├── upload_manifest.json               # SHA-256 hashes of all 453 files
├── EXCLUDED_FILES.md                  # excluded-file inventory and reasons
├── LICENSE_PROVENANCE.md              # license/provenance verification
├── TAKSH_V141_FINAL_QUALIFICATION_REPORT.md  # qualification evidence
├── taksh/                             # source package (6 files)
├── serving/                           # serving code (13 files)
├── evaluation/                        # evaluation code (55 files)
├── tokenizer/                         # tokenizer CODE only (train_tokenizer.py; .json vocab EXCLUDED)
├── training/                          # training code (65 files)
├── configs/                           # configs (45 files)
├── data/                              # data pipeline CODE only (.py files; NO data artifacts)
├── scripts/                           # scripts (36 files)
├── tests/                             # test suite (108 files)
└── docs/                              # documentation source (rst + md)
```

### Proposed Hugging Face repository
`ajit-ai/taksh` (consistent with existing repo `https://github.com/ajit-ai/Quantsmind-LLM`)

## 4. Publication accuracy (disclosures)

| Disclosure | Value |
|---|---|
| Hybrid independent functional qualification | 39/40 |
| C3 binding qualification | 28/28 |
| Neural model-only capability | NOT QUALIFIED (0 in all suites) |
| Hybrid deployment status | CONTROLLED PILOT ONLY |
| Remote GitHub Actions CI | NOT RUN (monthly usage exhausted) |
| Local CI qualification | 1,649 passed, 1 skipped, 0 failed |
| Production/GA claim | NOT made |

All capability claims are drawn from `TAKSH_V141_FINAL_QUALIFICATION_REPORT.md` and `qualify_v123.py` output. Hybrid results are not described as standalone neural model results.

## 5. Local validation

### Packaging/build/install
- `pyproject.toml`: `name = "taksh"`, `version = "1.0.0"` (dynamic from `taksh.__version__`), `license = "Apache-2.0"`
- Wheel builds successfully (`pip wheel . --no-deps --no-build-isolation`)
- Wheel contains 212 files, no checkpoints/credentials/temp files
- Isolated venv install: `import taksh` → `__version__ == "1.0.0"`; `import evaluation.operation_synth` OK
- `taksh --version` → `Taksh 1.0.0 (final)`, exit 0; unrecognized args exit 2 clean
- `import serving.openai_server` → routes `/v1/models`, `/v1/chat/completions` present, `/load` absent
- Legacy servers: no `/load` route; loopback default enforced

### CLI/API smoke tests
- `--version`: exit 0, prints version
- Invalid args: exit 2, clean argparse error, no traceback

### Integration workflow
- Hybrid end-to-end via `scripts.qualify_v123.py`: all invariants met
- Planner statuses: VALIDATED/UNSUPPORTED_PLAN/INVALID_PLAN distinctions preserved
- Unsupported planner coverage never treated as proven-invalid

### Security review
- Checkpoint loader: `weights_only=True` default; `weights_only=False` only via explicit `allow_unsafe_trusted=True`
- Serving binds `127.0.0.1` by default; `--allow-remote` opt-in
- Optional `QM_API_KEY` bearer auth
- No `POST /load` endpoints; no secrets in package
- No telemetry or network exfiltration in hybrid path

### Documentation examples
- All README code examples are syntactically valid Python
- Qualification results match `TAKSH_V141_FINAL_QUALIFICATION_REPORT.md`

### Asset hashes
All checkpoint, tokenizer, and suite digests verified unchanged (see §8 below).

## 6. Hugging Face authentication and publication

- `huggingface_hub` 1.28.0 is installed
- `huggingface_hub.hf_api` is importable
- `~/.cache/huggingface/token` does **not** exist — no HF authentication configured
- `huggingface-cli` availability not independently verified (no `huggingface-cli` command found on PATH)

**No Hugging Face repository was created. No files were uploaded. No repository visibility was changed. No model or Space was published.** Upload requires explicit owner authorization.

## 7. Deliverables

| Deliverable | Location | Status |
|---|---|---|
| Hugging Face publication directory | `taksh-hf-publication/` | Complete (453 files) |
| README/model card | `taksh-hf-publication/README.md` | Complete (HF metadata + disclosures) |
| Upload manifest with SHA-256 | `taksh-hf-publication/upload_manifest.json` | Complete (453 files) |
| Excluded-file inventory | `taksh-hf-publication/EXCLUDED_FILES.md` | Complete |
| License/provenance verification | `taksh-hf-publication/LICENSE_PROVENANCE.md` | Complete |
| Local publication validation | This report (§5) | Complete |
| Final GO/NO-GO recommendation | §8 below | NO-GO (awaiting authorization) |

## 8. Preservation verification (post-campaign)

- Checkpoint SHA-256: `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac` ✓
- Tokenizer SHA-256: `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988` ✓
- Suite digest: `1f5024e26c37a1186db5a924e5be1dbdafbfcc1e44bc8002849ad5be045cd419` ✓
- Package version: `1.0.0` ✓
- HEAD: `a654e5d` ✓

## 9. Remaining security and operational risks

1. **Remote CI unverified**: `ci.yml` jobs cannot be executed without a push (not authorized). Local equivalent covers all runnable steps.
2. **No HF token configured**: `~/.cache/huggingface/token` does not exist. Owner must configure authentication before upload.
3. **Legacy checkpoint support**: `server_security.load_trusted_state` and `openai_server.load_state` pass `allow_unsafe_trusted=True` for legacy checkpoints — explicit trusted boundary, not silent fallback.
4. **Model-only capability unqualified**: 0 functional across all suites; hybrid remains CONTROLLED PILOT ONLY.
5. **PR #1 remains open** with pre-existing CI adjudications; not part of this campaign.

## 8. Final GO/NO-GO recommendation

**NO-GO (awaiting owner authorization).**

The publication directory is upload-ready with all artifacts, hashes, and disclosures in place. However:
- No HF authentication is configured
- No Hugging Face repository has been created
- No files have been uploaded
- Owner authorization is required before any upload or publication

**This is NOT a publication.** All artifacts are preserved locally for owner review. Upon explicit authorization, the owner may:
1. Configure HF authentication (`huggingface-cli login`)
2. Create the Hugging Face repository (`ajit-ai/taksh`)
3. Upload the publication directory contents
4. Set repository visibility (public)
5. Tag and release the package

**Implementation verification: COMPLETE.**
**GA readiness: NOT YET GA-READY** (pre-existing conditions; not a regression from this campaign).
**Publication: NOT PERFORMED — awaiting explicit owner authorization.**
