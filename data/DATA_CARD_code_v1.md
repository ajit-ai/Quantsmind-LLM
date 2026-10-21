# Data Card — code_corpus_v1 (external, licence-verified)

Created: 2026-10-15 · Pipeline: `data/build_code_corpus_v1.py` · Sources: `configs/data/approved_code_sources.yaml`
Tokenizer used for the token count below: `tokenizer/qm_code_bpe_v0.json`

## Contents

| Field | Value |
|---|---|
| Records (chunks) | 447 |
| Tokens (own code BPE, 5,822 vocab) | 2,277,473 (avg 5,095/record) |
| Characters | 6,701,042 |
| Split | train 437 · val 8 · test 2 (content-hash, no seed) |
| Sources | 9 repositories, all pinned to a full commit SHA |
| Licences | MIT 325 · Apache-2.0 82 · BSD-3-Clause 40 |
| Manifest | `data/code_corpus_v1.manifest.json` (`status: complete`) |

Per source (records kept):

| Source | Repo | Licence | Records |
|---|---|---|---|
| psf-requests | psf/requests | Apache-2.0 | 19 |
| pallets-flask | pallets/flask | BSD-3-Clause | 23 |
| psf-black | psf/black | MIT | 37 |
| urllib3-urllib3 | urllib3/urllib3 | MIT | 39 |
| pallets-click | pallets/click | BSD-3-Clause | 17 |
| pytest-dev-pytest | pytest-dev/pytest | MIT | 79 |
| aio-libs-aiohttp | aio-libs/aiohttp | Apache-2.0 | 63 |
| pypa-setuptools | pypa/setuptools | MIT | 135 |
| tiangolo-fastapi | tiangolo/fastapi | MIT | 35 |

## Provenance and licensing

* Every source is enumerated in `configs/data/approved_code_sources.yaml`. There is
  no discovery or search path in the pipeline; a repository that is not in that file
  is never fetched. The legacy search-based builder (`data/build_code_corpus.py`)
  now refuses to run unless explicitly overridden.
* Each source records repo, full 40-character commit SHA, declared SPDX id, attribution
  string and retrieval date.
* **Licence verification is a blocking build gate.** `data/verify_licenses.py`
  reads the actual licence file at the pinned commit, hashes it (SHA-256), classifies
  the family, and matches it against both the pinned SPDX id and the allowlist
  (`mit`, `apache-2.0`, `bsd-3-clause`, `bsd-2-clause`, `isc`). `NOASSERTION` is
  refused. `data/build_code_corpus_v1.py` refuses to ingest a source unless the
  verification artifact contains an acceptable result **for that exact commit**.
  Missing or failed evidence blocks the source; it is never treated as permission.
* **Nested licences.** A licence file in a subdirectory governs that subtree, not the
  whole repository and not nothing. Exclusion prefixes are *derived* from where
  licence files actually are: a subtree declaring a family different from the root,
  or one whose text cannot be classified, is excluded along with its descendants.
  A subtree with no licence file is not assumed to conflict — absence of a file is
  not a licence claim.
* Two exclusions are in force as conservative project policy, pending human review:
  `psf/black` → `src/blib2to3/` (GPL-2.0) and `pypa/setuptools` →
  `setuptools/config/_validate_pyproject/` (AGPL-3.0 text). Both are recorded in the
  manifest with their reason and the pinned revision is unchanged.
* Every retained record carries `repo`, `commit`, `spdx`, `attribution`, `file_path`,
  `retrieved_at` and `content_sha1`.
* Vendored third-party trees (`_vendor`, `vendor`, `third_party`, `node_modules`,
  `site-packages`) are excluded, because attributing the pinned repository's licence to
  someone else's vendored code would be wrong.

## Filtering

Applied per file, with the reason recorded in the manifest rather than silently dropped:

| Rule | Result |
|---|---|
| Path exclusions (tests, docs, examples, tutorials, vendor, scripts, benchmarks) | 2,126 rejected pre-download |
| Below 200 bytes / above 256 KiB | 37 too small, 4 too large (oversize refused from the tree listing, never transferred) |
| Alpha ratio < 0.55 | 4 |
| `@generated` / "do not edit" markers | 0 |
| Does not parse as Python (`compile`) | 0 in the accepted set |

Syntax validity is a *parse* result only. It says nothing about code quality, and no
downloaded file is ever executed.

## Deduplication and splits

* Exact: SHA-1 of NFKC-normalised text with trailing whitespace stripped. 0 duplicate
  records remain.
* Near-duplicate: MinHash LSH, 64 permutations, 16 bands, 0.8 Jaccard over 8-word
  shingles. Applied before splitting.
* Split assignment is a pure function of content:
  `bucket = int(sha1(text)[:8], 16) / 0xFFFFFFFF`, then train / val / test by the
  configured fractions. It needs no random seed and does not depend on collection order.
* Measured leakage on the delivered shard: 0 chunks with the same content hash in two
  splits, 0 surviving near-duplicate pairs, 0 near-duplicate bands crossing a split.

## Known limitations

* **Not decontaminated against published benchmarks.** An audit against the
  benchmark data actually present in this repository (10 hand-transcribed
  problems in `evaluation/eval_code.py`) found no overlap. The **published**
  suites — HumanEval (164) and MBPP (500) — are **not present here and were not
  checked**, so this corpus is **not** demonstrated to be free of
  published-benchmark overlap. See `data/contamination_audit.json`. Detection is
  lexical only; no semantic decontamination is claimed.
* **Two licence conflicts need a human decision.** `psf/black` is MIT at the root
  but `src/blib2to3/LICENSE` is GPL-2.0, and `pypa/setuptools` is MIT at the root
  but `setuptools/config/_validate_pyproject/` contains AGPL-3.0 text. Those
  subtrees are quarantined from the corpus rather than relabelled. See
  `data/licence_verification.json`.
* **Licence verification is text-based, not legal advice.** It reads the licence
  file at the pinned commit, hashes it, classifies the family and compares it
  with the pinned SPDX id and the allowlist. A human must confirm obligations
  before redistribution.
* **Holdouts are thin.** Source-group splitting assigns whole repositories. With
  9 approved repositories the holdouts hold 1-2 repositories each. They are
  deterministic and reproducible but are **not statistically sufficient** for
  model evaluation and **not benchmark-independent**.
* **Split is per-repository**, not per-chunk, so a single repository dominates
  whichever split it lands in.
* **Near-duplicate detection is lexical only.** It finds near-identical text, not
  paraphrases, renamed symbols or reimplementations.
* **PII scrubbing is token-aware and narrow.** Only string and comment tokens are
  scrubbed; numeric literals and identifiers are never rewritten. Patterns are
  India-specific (Aadhaar, PAN) plus a 10-digit phone heuristic and email. Names,
  addresses, IPs, JWTs and connection strings are **not** detected, so personal
  data outside those patterns reaches the corpus.
* One repository, `sphinx-doc/sphinx`, was rejected during verification
  (`NOASSERTION`) and is recorded in the config's `rejected:` list so the decision
  is auditable rather than invisible.

## Reproduce

```powershell
python -m data.build_code_corpus_v1 --include-held-back --max-repos 9
```

Verify the manifest and shard independently:

```powershell
python -m pytest tests/test_code_corpus_v1.py -q --basetemp=$env:TEMP\opencode\pytest_bt
```

Re-running the same configuration against the same pinned commits reproduces the same
records and the same split assignment; this was verified against the live network by
re-running three sources and comparing content hashes and splits record-for-record.
