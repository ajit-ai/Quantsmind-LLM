# TAKSH — Release Governance

**Document ID:** `TAKSH-RELEASE-GOVERNANCE`
**Version:** 1.0 — established 2026-11-11 by milestone L0.7.6
**Applies to:** all TAKSH releases from v1.0.0 through v3.0.0 (and any pre-release suffixes thereafter)
**Authority:** release-qualification and release-authorization rules. Binding on engineering; binding on any agent; overrideable only by the owner in writing.

---

## 1. Versioning policy

TAKSH uses package version `MAJOR.MINOR.PATCH` for the `taksh` package, and its own milestone-granularity notation (`L0.x.y`) for development milestones. Development milestones produce *package* version increments only when a release is actually cut; until then the package version is **not bumped**.

Rules:

1. `0.x` — development series. Never published as a stable release guarantee; milestone-only.
2. `1.x.y` — the v1.x series ("Independent Adaptive Coding Assistant" lineage). `v1.0.0` is the first GA candidate gate (L1.0.0 milestone).
3. `2.x.y` — the v2.x series ("Repository-Aware" lineage). `v2.0.0` at L2.10.
4. `3.x.y` — the v3.0 series ("Advanced Autonomous" lineage). `v3.0.0` at L3.7.
5. Pre-releases (`-rcN`, `-alpha`, `-beta`) may be cut **only** inside an owner-authorized release milestone, and never republished in place.
6. Package version `1.0.0` does **not** encode any milestone gate. Roadmap progress is not implied by package version arithmetic.

Milestone numbering (`L0.x.y`) is decoupled from package version; never infer one from the other.

---

## 2. Release qualification requirements (every release)

A release is not complete because implementation or tests pass. Every TAKSH release ships:

1. **Exact starting and ending commit SHA** (recorded in the release record).
2. **Branch and remote verification** (`git ls-remote` evidence at publication time).
3. **Change inventory** — file-level list with provenance classes (new / modified / deleted).
4. **Architecture and compatibility review** — eight-subsystem integrity, DLCAE preservation, canonical-configuration integrity, frozen benchmark untouched.
5. **Focused test evidence** — the milestone's own qualification suite result; never merely "the suite was green in CI at some point".
6. **Regression evidence** — inherited regression suite (L0.6.0/R1/L0.6.1/…) still green at release commit.
7. **CI evidence and explicit limitations** — what CI ran, what it skipped (e.g., the untracked-artifact tests), what could not run and why.
8. **Frozen benchmark results** — unchanged benchmark identity; scores published with the exact task-set hash; 0/30 stands until honestly displaced.
9. **Security and safety assessment** — no execution of untrusted code; no network access introduced without owner approval; checkpoint can't reach the network; API auth paths audited.
10. **Resource measurements** — memory/VRAM/disk/threads, with measured-vs-estimated separation.
11. **Known issues and deferred work** — explicit list; nothing silently dropped.
12. **Explicit capability claims with evidence** — every claim maps to a frozen executable benchmark row, or it is not claimed.
13. **Release decision** — owner-recorded decision with a date and a reference.

---

## 3. Status vocabulary (binding)

| Status | Meaning |
|---|---|
| NOT_STARTED | no engineering or evidence exists; may not be counted as done in any claim |
| IN_PROGRESS | bounded qualification underway; milestone output not accepted until gate passes |
| BLOCKED | cannot proceed; blocker unresolved; may not be reported as qualified/GA |
| CONDITIONAL_PASS | qualification passed with documented, bounded conditions; owner must accept them explicitly |
| QUALIFIED | all milestone acceptance criteria met with evidence; owner sign-off recorded |
| RELEASE_CANDIDATE | release candidates built and verified; awaiting owner GA decision |
| GA | general availability released under owner authorization and publication grant |
| DEFERRED | intentionally postponed by owner decision (documented, never silently) |

**No status may be upgraded merely because a document, interface, or test exists.** Machine registries (`configs/roadmap/taksh-v3-roadmap.yaml`) record statuses exactly as qualified; documentation restates them; neither invents them.

---

## 4. Evidence thresholds

### 4.1 Frozen benchmarks (unchanged identities)

| Artifact | Identity | Rule |
|---|---|---|
| Frozen Python benchmark | `evaluation/l02_python_bench_v1.json` · SHA-256 `854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6` | tasks, expected outputs, scoring thresholds are frozen; never altered for a pass |
| Multilingual benchmark | `evaluation/l051_multilingual_benchmark_v1.json` · SHA-256 `7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e` | 64 tasks; denominator fixed at ≥ 16 executed tasks AND ≥ 50% pass ratio per language |
| Karkain benchmark | `reports/l060r1_kark_bench_schema.json` | schema only; **0 tasks instantiated**; Karkain remains unqualified |

### 4.2 Language qualification

A language is FUNCTIONALLY_QUALIFIED or ENGINEERING_QUALIFIED only when:

- ≥ 16 executed tasks from the frozen multilingual benchmark for that language pass at the frozen threshold, AND
- at least 50% of those executed tasks pass, AND
- a language-level report exists, reproducible by any qualified reviewer.

Registration in the 53-language catalogue, presence in the tokenizer vocabulary, documentation, or `.kark` mapping never implies capability.

### 4.3 Milestone acceptance

A milestone is not accepted because: tests pass, a report exists, a doc is consistent, a number improved once, or a prior gate passed. Acceptance requires the milestone's own gate with its own declared evidence, plus the inherited regression suite green at the same commit.

---

## 5. Authorization boundaries

| Activity | Owner approval required |
|---|---|
| Training or fine-tuning any model | explicit, in writing, per run |
| Acquire or download external data | explicit, per candidate, legal review first |
| Provision or spend on compute/APIs/storage | explicit, with budget and residency |
| Tag or release any version | explicit, per release |
| Merge to `develop` or `main` | explicit, per merge |
| Create or modify `TAKSH-KARK-BENCH` contents | explicit, per gate milestone |
| Change frozen benchmark contents | prohibited outside owner-authorized milestone |
| Publish any artifact (PyPI, HF, docs site) | explicit, per artifact |
| Enable CI on default branch | explicit, per workflow change |
| Change release governance itself | explicit, in writing, recorded in `TAKSH-ROADMAP-CHANGELOG.md` |

Engineering never authorizes any of these. Documentation never authorizes any of these. The blank owner decision record cannot authorize anything.

---

## 6. Release-decision rule

A TAKSH release is **rejected** by default. It becomes releasable when, in order:

1. Every blocking prerequisite is recorded as discharged with evidence.
2. The release-qualification record (§2) is complete.
3. Frozen benchmarks pass at the declared thresholds, or honesty is recorded (a failed benchmark does not ship as a pass — it ships as a documented disappointment).
4. The owner records an explicit release decision with a date.
5. Tags and publication occur **only** under that recorded decision.

If the owner does not record a decision, the release does not happen.

---

## 7. Release evidence record (minimum shape)

```yaml
release: vX.Y.Z
date: "YYYY-MM-DD"
commit: <sha256>
branch: feature/l060-foundation-model-qualification
remote_verification: "git ls-remote evidence"
milestones_satisfied: [L…]
frozen_benchmark:
  python: {hash: "854c7ca8…", result: "<n>/30"}
  multilingual: {hash: "7a839060…", per_language: {…}}
regression_evidence: [suite ids…]
ci_limitations: "explicit list"
security_assessment: "summary + report link"
capability_claims: ["each claim ↔ benchmark row"]
known_issues: […]
owner_decision: "<reference>"
```

This record is filed under `docs/releases/` (created only in a release milestone).

---

## 8. GA gates

GA additionally requires:

- Frozen Python benchmark threshold met (owner-approved bar), or honest failure recorded; GA impossible if it is not addressed.
- Multilingual qualification for every language claimed at GA.
- Package integrity verified at installation time (fresh environment).
- Reproducible build of the release artifact.
- Owner-signed release decision.

---

## 9. Change control for this document

Any change to this file requires: a reason, a diff reviewed by the owner, a changelog entry in `docs/TAKSH-ROADMAP-CHANGELOG.md`, and explicit owner approval recorded in writing. Silent edits are invalid and must be reverted.
