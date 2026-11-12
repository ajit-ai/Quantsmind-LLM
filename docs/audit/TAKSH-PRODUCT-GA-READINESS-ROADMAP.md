# TAKSH — Product GA Readiness Roadmap (L1.0.0, 2026-11-12)

**Authorization:** A1.1-C (preparation-only) / A2-E (no compute). Nothing in this roadmap authorizes training, acquisition, spend, release, or merge.
**Position of the TAKSH program today:** v1.0.0 GA **NOT established**; canonical model **MISSING**; capability **NOT_EVALUATED**; Karkain **DISCOVERED**; benchmark 0/30 stands.

This roadmap deliberately separates three axes that are routinely conflated:

- **Engineering readiness** — infrastructure and honesty gates exist, work, and fail closed.
- **Model readiness** — a qualified canonical artifact exists and runs deterministically.
- **Demonstrated coding capability** — frozen benchmark measurements ≥ agreed bar, reproducible, documented.

Stages are ordered by dependency. No stage authorizes the next. Every stage has entry/exit criteria and explicit owner actions.

---

## Stage 0 — Baseline & honesty (DONE / partial)

- ✓ L0.6.x, L0.7.x, L0.8.0-PRE, L1.0.0-ENG/ENG2/ENG3 completed on feature branch.
- ✓ Isolated CI recovery integrated into develop (PRs #2/#3; develop CI green; merged to main by owner 2026-11-12).
- ✓ Owner decisions A1.1-C / A2-E enforced everywhere.
- ✓ Frozen benchmark 0/30 preserved.
- ✗ No canonical artifact; no capability evaluated; no release claim.

## Stage 1 — Integration of the engineering core into develop (PENDING owner)

Objective: move the qualified DLCAE stack from the feature branch to develop without disturbing develop or main.

Entry criteria:
- this milestone's integration plan reviewed and accepted;
- owner authorizes integration PRs;
- each PR is independently scoped and mergeable.

Deliverables:
1. PR: roadmap registry + release governance docs (cherry-picked `84888e6` subset) — docs only.
2. PR: L0.6.0-R1 DLCAE architecture & registry (cherry-picked `45f0dc9`) — docs + validators + tests.
3. PR: L0.6.1 DLCAE-aware training pipeline (`1fef89d`).
4. PR: L0.6.2 inference qualification (`9721aa6`).
5. PR: DLCAE runtime (`efcda29`).
6. PR: semantic validation foundation (`6065394`).
7. PR: end-to-end integration (`d2f0d14`).

Exit criteria:
- every PR: focused tests green locally + on develop CI;
- `develop` CI remains green;
- frozen benchmark hashes intact (BLOCKED_ARTIFACT_ABSENT on develop handled by porting artifacts only with owner sign-off);
- no training runs anywhere, no tags, no releases.

Owner action required: yes (7× green light + review). Blocks: Stages 2–6.

## Stage 2 — Canonical artifact existence (BLOCKED)

Entry criteria:
- authorized training data (owner decision; not met);
- authorized training environment (A2 decision; not met);
- A1.1-A training run authorized (not met);
- training corpus governance completed (L0.8.0 entry contract; pending owner).

Deliverables:
- trained `taksh-foundation-111m` artifact;
- provenance manifest (code revision, snapshot ids, seeds, token accounting);
- artifact SHA-256 + config/tokenizer fingerprint verification.

Exit criteria:
- artifact loads, passes fingerprinting and geometry checks;
- artifact loads on at least one host with documented memory profile;
- frozen benchmark preserved; capability still NOT claimed.

Owner action required: written authorization (A1.1-A-class) + environment approval (A2-class) + data corpus decision. **Currently blocked.**

## Stage 3 — Deterministic inference qualification (BLOCKED stage 2)

Entry criteria: Stage 2 complete + tokenizer audit trail intact.

Deliverables:
- reproducible generation (seeded) end-to-end from cold start;
- trace records (`taksh.inference_trace/1`) for every evaluation run;
- documented memory/latency envelope (CPU host of this class, optional GPU if authorized).

Exit criteria:
- identical outputs across two cold runs with identical seeds;
- no silent fallback to any other model;
- full artifact lineage retained.

Owner action required: none beyond Stage 2 authorizations (inference is read-only).

## Stage 4 — Capability measurement (BLOCKED stages 2–3)

Entry criteria: stages 2 & 3 done + frozen benchmarks ported to develop with owner sign-off + evaluation harness qualified.

Deliverables:
- `reports/capability_eval/<date>/` per `TAKSH-CODING-CAPABILITY-EVALUATION-PLAN.md`;
- baseline measurement with n ≥ 3 seeds;
- model-only vs tool-assisted split reporting.

Exit criteria:
- honest measurement published as an internal report only (no public release yet);
- owner reviews; GA decision unaffected today.

Owner action required: explicit green-light for capability measurement (owner treats this as the checkpoint for any claim).

## Stage 5 — Capability improvement (BLOCKED stage 4)

Entry criteria: Stage 4 measurements; owner aligns on improvement roadmap (SFT, RL, additional data — none authorized yet).

Deliverables: improved benchmark results with same seeds/harness; no regression on frozen set; DLCAE/RAG/hybrid comparisons performed if authorized.

Exit criteria:
- documented, reproducible delta ≥ owner-agreed bar;
- contaminated runs rejected outright;
- no public claim.

Owner action required: capability-tuning authorizations (not yet granted).

## Stage 6 — Multilingual & Karkain qualification (BLOCKED hard)

Entry criteria:
- Karkain host/CI blocker resolved (`kcc.exe` `0xC0000135` or documented workaround, owner-approved);
- Karkain corpus license cleared (owner);
- TAKSH-KARK-BENCH ≥ 1 instantiated task (owner authorizes design);
- harness + qualification bar written and frozen; owner approves bars.

Deliverables:
- Karkain COMPREHENSION→FUNCTIONALLY_QUALIFIED journey through DLCAE;
- KARK-BENCH run with at least 1 passing task;
- multilingual benchmark re-run row for Karkain.

Exit criteria:
- KARK-BENCH pass with recorded evidence; model-only vs compiler-assisted reported separately;
- KARK-BENCH instantiation count ≥ 1; Karkain remains DISCOVERED until then;
- no multilingual capability claim beyond the KARK-BENCH result;
- Karkain repository/specs/compiler untouched by TAKSH.

Owner action required: Karkain track authorization; host repair; separate owner decisions.

## Stage 7 — Safety, privacy and operational readiness (NOT STARTED)

Entry criteria: stages 1–6 progressed; product scope defined (release or internal).

Deliverables:
- sandboxing/runtime policy audit (current design: tool execution only in isolated workspaces; no network; no untrusted projects);
- red-team report; secret hygiene/audit; privacy & data-handling statement (owner-facing, not public);
- monitoring/observability for deployed instances; safe-failure semantics (model present or clear failure — no fake outputs).

Exit criteria:
- owner-accepted safety report; no critical findings open; deployment boundaries documented.

Owner action required: yes (scope and publication of any safety artifacts).

## Stage 8 — Release candidate (BLOCKED stage 7)

Entry criteria: stages 1–7 complete.

Deliverables:
- RC manifest (code, artifact, benchmarks, reports); RC verification runbooks; rollback procedure;
- release governance checklist completed; owner signs release decision record.

Exit criteria:
- everything reproducible from source; rollback demonstrated; no silent unverifiable claims anywhere in documentation.

Owner action required: yes (final release decision).

## Stage 9 — Public GA (NOT STARTED)

Entry criteria: stages 1–8 complete + explicit owner authorization + external distribution decision (publication, HF, PyPI, or private).

## Exit definition (product GA)

- canonical artifact exists & traceable; inference reproducible;
- Python benchmark measured with 3 seeds and honestly reported (capability claim granted or refused by evidence);
- governance, safety, privacy signed off by owner;
- documentation claims match reality at every point;
- no tag/release/publication without an explicit owner decision.

## Immediate actions for the owner

1. Review the integration PR sequence (Stage 1) and authorize cautiously — or instruct re-planning.
2. Decide the status of the frozen benchmark and corpus (blocked or port).
3. Decide on v1.0.0 → develop → main → public release strategy — currently not authorized.

## Karkain — the single most important blocked item

| Fact | Value |
|---|---|
| Language 54 status | `DISCOVERED` |
| KCC | `0xC0000135` on this host (Windows DLL missing) |
| KARK-BENCH | 0 instantiated tasks |
| Karkain corpus | 0 authorized tokens |
| TAKSH-KARK-BENCH bars | undefined ("NOT DEFINED YET") |
| Owner action required | explicit Karkain authorization, artifact/runtime repair or different host |

Nothing in the TAKSH package replaces KCC. No Karkain capability claim can exist today; every verdict for Karkain tool-assisted work is **BLOCKED**.
