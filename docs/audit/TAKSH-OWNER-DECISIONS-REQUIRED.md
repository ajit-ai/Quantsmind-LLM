# TAKSH — Owner Decisions Required (L1.0.0, 2026-11-12)

**Authorization:** A1.1-C / A2-E. Nothing in this document is a decision recorded; it is the list of decisions the owner must supply before TAKSH can leave its current state.

Format: every entry has — **Decision** · **Status today** · **Needed for** · **Exact wording required**.

---

## 1. Training authorization (A1.1-class)

| Decision | Status | Wording required |
|---|---|---|
| A1.1-A bounded research training authorization | **NOT GRANTED** | "A1.1-A authorized on YYYY-MM-DD: train `taksh-foundation-111m` from the L0.8.0 entry contract under owner-approved data and environment." |
| Reaffirm A1.1-C (preparation-only) or reopen A1.1-A | Current: A1.1-C. The record exists on feature branch `docs/L075_L080_OWNER_DECISION_RECORD.md`; absent on develop/main | Re-affirm explicitly on develop: "A1.1-C continues; no training acts" or supersede |
| Warm-start weights for the canonical run | No authorization for `v180_sft` or any substitution | Explicit decision needed: fresh init (default) or warm-start (non-standard; would require vetting) |

## 2. Compute authorization (A2-class)

| Decision | Status | Wording required |
|---|---|---|
| A2-A existing local hardware only | Unselected | "A2-A authorized…" |
| A2-B owner-provided GPU | Unselected | named hardware + preflight plan |
| A2-C external compute investigation | Unselected | "research allowed; no account/spend" |
| A2-D external compute execution (spending budget) | Unselected | explicit budget, region, provider, quota |
| A2-E reaffirm (no compute) | **CURRENT** | treat as the default until re-opened |

Every A2 option above requires explicit writing; silent defaults do not count as decisions.

## 3. Data authorization

| Decision | Status | Wording required |
|---|---|---|
| Python corpus licensing — the L0.7.0/L0.7.1 records ratify only *internal research use*; nothing else | research-only frozen | "OK for internal research training run only" (as required by L0.7.0 rules) |
| Any public/training corpus redistribution | unknown/unauthorized | must be explicit and scoped; nothing is authorized for publication |
| Karkain corpus authorization | **not authorized; 0 tokens** | "Karkain corpus stays unauthorized until owner re-opens" |
| Multilingual corpus expansion | blocked | fresh counsel review needed for any beyond Python research scope |

## 4. Integration into develop / main

| Decision | Status | Wording required |
|---|---|---|
| Authorize the L0.6.x→L1.0.0-ENG3 integration PR sequence (integration PRs 1–9 from the commit matrix) | not started; proposed | "integration sequence authorized" — cherry-pick series only |
| Cherry-pick `84888e6` (roadmap docs/registry) | not authorized by this milestone; not integrated | "Roadmap PR #1 authorized" |
| Frozen benchmarks to develop (needed for capability measurement) | BLOCKED_ARTIFACT_ABSENT on develop | "Frozen benchmarks may land on develop via a dedicated PR (owner decision) or remain feature-branch only" |
| Port `evaluation/l051_language_registry.py` + `configs/feasibility/l051_languages.yaml` (untracked) | untracked owner artifacts — no branch | "Declare these owner-repo-controlled files part of the integration surface" (requires importing untracked files — explicit owner consent) |
| Merging anything into `main` | **owner action only**; this milestone does none; develop→main merge happened 2026-11-12T04:56Z by owner, outside this milestone's scope | owner's written decision each time; no silent batches |

## 5. Capability & release decisions

| Decision | Status | Wording required |
|---|---|---|
| Capability measurement (Python benchmark 0/30 baseline) | not authorized; never executed under this program | "capability evaluation authorized …" with seeds/harness as in the eval plan |
| External capability-claim language | forbidden | no marketing claims without evidence + release decision |
| v1.0.0 GA promotion | **NOT ESTABLISHED** | requires: canonical artifact + measured capability ≥ owner bar + safety sign-off |
| Publication channel | none chosen | HF/PyPI/private-vs-public → explicit owner sign-off |
| Karkain as separate first-class language product | DISCOVERED only | re-open only after Stage 6 prerequisites |

## 6. Safety, privacy, sandbox authorizations

| Decision | Status | Wording required |
|---|---|---|
| Tool-execution policy (isolated workspaces; no network; untrusted code is never executed outside sandboxes) | current default design | owners must sign off before GA |
| Human-approval gates for any action that modifies user files | enforced by design (suggestion-only repair; never auto-apply) | must remain true for any product release |
| Data privacy statement (training data, provenance, licensing) | not yet written | owner-approved before any publication-oriented release |

## 7. Karkain track (if the owner wishes to start it)

| Decision | Status | Wording required |
|---|---|---|
| KCC `0xC0000135` resolution strategy (host repair / different host / owner-provided Windows build) | blocked | "Karkain track unblocked by " (choose) |
| TAKSH-KARK-BENCH instantiated tasks | **0** | "KARK-BENCH task instantiation authorized; bar: " (choose) |
| Karkain corpus license (internal or research-only at this stage) | unauthorized | written authorization |
| Karkain SPEC pin | v1.1.0 unchanged | change only with owner authorization |
| KARK-BENCH authority boundary (never bypass KCC) | permanent | restated in every release decision |

## 8. Infrastructure & CI decisions

| Decision | Status | Wording required |
|---|---|---|
| `TAKSH_ENABLE_TRAINING_CI` variable | **MUST KEEP UNSET** (A1.1-C / A2-E) | owner must personally set if future training authorized — and no one else may |
| Python 3.11 runtime qualification | not executed anywhere yet | "Run 3.11 CI" or "declare host-supported = 3.14, 3.11 unverified" decision |
| Full CI on develop/main now required for every future merge | already rule; currently satisfied | keep required-checks green; do not tweak |
| `v1.0.0` Git tag — already published 2026-10-21 | pre-existing | do not retag; do not create new tags here |

## 9. Summary: what the owner must say, in order, to leave the frozen state

1. "I authorize integration PRs #1..#9 per the integration plan (or: I veto one of them)."
2. "I keep A1.1-C, or: I reopen A1.1-A for bounded training with ref <ID>."
3. "Compute: no option / A2-A / A2-B / A2-C / A2-D with budget …."
4. "Frozen Python benchmark: remains 0/30 historical evidence; or: I authorize a re-run as part of capability measurement."
5. "`v180_sft`: remains a frozen historical artifact; or: demoted per owner decision."
6. "Frozen benchmarks: port to develop in a dedicated PR / leave feature branch only."
7. "v1.0.0 GA: not today (or: schedule GA review after my decisions on 1–6 land)."
