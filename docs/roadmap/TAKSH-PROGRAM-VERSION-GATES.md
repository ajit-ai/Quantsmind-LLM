# TAKSH Program Version Gates

**Status:** CONTROL DOCUMENT — no release, tag, merge or publication is
authorized by this document.

**Machine-readable inputs:**
- `configs/roadmap/five_version_program.yaml`
- `reports/taksh_version_gates.json`
- `python -m evaluation.version_gates`

---

## 1. Why this document exists

Three numbering schemes are in use:

| Namespace | Meaning | Current example |
|---|---|---|
| **Package version** | Installable/published Taksh package. Authoritative source is `taksh.__version__`. | `1.0.0` |
| **Master roadmap** | Approved long-range forward-development index. Remains canonical. | `v1.2.0` — Python data and tokenizer foundation |
| **Program label** | Execution label for one of the five consolidated implementation slices. Not a package version. | `1.2.0` — Native Model Capability |

The program labels collide numerically with master-roadmap versions but have
different responsibilities. They must not be used interchangeably.

**Worked example:** program label `1.2.0` means *Native Model Capability*
(training readiness and a canonical checkpoint). Master roadmap `v1.2.0` means
*Python data and tokenizer foundation*. These are not the same milestone.

---

## 2. Release rule

1. The package stays at `1.0.0` until an authorized release.
2. A program label does **not** automatically become a package version.
3. No tag, GitHub release, PyPI upload, Hugging Face upload or merge to
   `develop`/`main` may use a program label as justification.
4. At release time, the owner selects the package version, and
   `taksh.__version__`, changelog, release notes and
   `python scripts/check_version_consistency.py` are updated together.
5. A release requires every applicable gate below to be `PASS` and every
   authorization flag to be `true`.

The current gate report says:

- package version: `1.0.0`
- release ready: **false**
- verdict: **NO_RELEASE_AUTHORIZED**
- program-label release versions: **none**

---

## 3. Program gate matrix

| Program label | Scope | Status | Package version | Artifact | Exit gate |
|---|---|---|---|---|---|
| `1.1.0` Native Foundation | Config, tokenizer binding, model/inference contracts | `PASS` | `1.0.0` | source/config only | Source contracts engineering-qualified; no model artifact claimed |
| `1.1.1` Native Inference Mechanics | Generation contract, safe loading, EOS handling | `PASS` | `1.0.0` | source/tests only | Mechanics pass focused tests; production inference remains blocked |
| `1.2.0` Native Model Capability | Training readiness, canonical checkpoint, measured model | `BLOCKED` | `1.0.0` | canonical `taksh-foundation-111m` checkpoint | Checkpoint exists, geometry/tokenizer pass, training evidence recorded |
| `1.3.0` Coding Intelligence | Orchestration, validation, specialists | `PARTIAL` | `1.0.0` | evaluation report | Frozen benchmark run against canonical checkpoint with raw outputs |
| `1.4.0` Autonomous Engineering | Multi-file engineering, tool calling, permissions | `PARTIAL` | `1.0.0` | qualified agentic surface | Tool calling and multi-file workflows qualified end-to-end |
| `2.0.0` Production GA | Security, deployment, licensing, release | `NOT STARTED` | `1.0.0` | GA release (not created) | Security, deployment, licensing and owner release gates all pass |

The master-roadmap relationship is indicative only and is recorded in the
machine-readable config. It does not renumber the master roadmap.

---

## 4. Current blockers

`1.2.0` remains `BLOCKED` by:

- training authorization (A1.1-A);
- compute authorization (A2);
- a training-authorized, licence-reviewed corpus;
- a canonical `taksh-foundation-111m` checkpoint.

`1.3.0` remains `PARTIAL` because no canonical checkpoint or frozen benchmark
artifacts are available for a measured evaluation.

`1.4.0` remains `PARTIAL` because tool calling and agentic multi-file editing
are not implemented or qualified.

`2.0.0` remains `NOT STARTED` because upstream capability, security,
deployment, licensing and release decisions are absent.

---

## 5. Status vocabulary

Only these statuses are valid:

- `PASS`
- `PARTIAL`
- `BLOCKED`
- `NOT STARTED`
- `NOT EVALUATED`

`PASS` for a source-only label does not imply model capability, production
readiness or publication authorization.

---

## 6. Release checklist (not executed)

- [ ] Owner selects the package version and records release authorization.
- [ ] All applicable program gates are `PASS`.
- [ ] `taksh.__version__` is updated.
- [ ] Changelog and release notes distinguish source/model/dataset readiness.
- [ ] `python scripts/check_version_consistency.py` passes.
- [ ] Focused and full test evidence is recorded.
- [ ] Source package contains no weights, data or checkpoints.
- [ ] Weight and dataset licences are resolved separately.
- [ ] Capability evidence is present and honest.
- [ ] Security and deployment gates are closed.
- [ ] Tag and release are created only after the owner authorizes them.
- [ ] Publication to PyPI/Hugging Face/GitHub Releases is separately authorized.

---

## 7. Non-actions

This control document does not:

- bump the package version;
- authorize training or compute;
- create a checkpoint or export;
- publish anything;
- merge this branch into `develop` or `main`;
- create a tag or release;
- change the master roadmap.
