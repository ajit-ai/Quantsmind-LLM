# TAKSH Design Architect — Feature Design

**Date:** 2026-11-14
**Status:** P0 design only. No capability is claimed. No model is trained. No external data is acquired.
**Direction:** coding-only genius; Design Architect is a coding-adjacent specialist, not a general design assistant.
**Authority:** subordinate to the TAKSH master roadmap and release governance.

---

## 1. Purpose

Design Architect converts software requirements and repository context into reviewable architecture artifacts. It is the front-end half of coding intelligence: it helps define what should be built before the coding engine implements it.

Design Architect does **not**:

- implement the design;
- execute generated code;
- mutate repository files;
- claim a finished architecture is correct without review;
- replace ADRs, threat modeling, security review or engineering approval.

Design Architect produces structured, inspectable artifacts only.

---

## 2. Audience and use cases

Primary use cases:

1. requirements-to-architecture proposal;
2. ADR generation and critique;
3. C4/Mermaid diagram generation;
4. API/data contract drafting;
5. migration and rollout planning;
6. repository-context-aware architecture review;
7. coding-agent handoff artifacts.

Target users: staff engineers, solution architects, platform teams, security reviewers and coding-agent operators.

---

## 3. Contracts

### 3.1 Prompt contract

`qm.architect_design/1`

Input structure:

```json
{
  "schema": "qm.architect_design/1",
  "requirements": [],
  "constraints": [],
  "repository_context": {},
  "languages": [],
  "non_functional": {},
  "requested_artifacts": ["overview", "adrs", "diagrams", "api_specs", "risks"],
  "review_level": "draft|review|approval_candidate"
}
```

Rules:

- explicit assumptions required when requirements are incomplete;
- open questions required when information is missing;
- every artifact carries `source_request_sha256`;
- no file paths in generated artifacts are treated as writes;
- output is Markdown + Mermaid + OpenAPI/AsyncAPI fragments, not executable code.

### 3.2 Artifact schema

`taksh.architect_artifacts/1`

```json
{
  "schema": "taksh.architect_artifacts/1",
  "summary": "string",
  "assumptions": [],
  "open_questions": [],
  "decisions": [{"id": "ADR-001", "title": "...", "status": "proposed", "decision": "...", "rationale": "...", "consequences": []}],
  "diagrams": [{"kind": "mermaid", "type": "C4Context|C4Container|sequence|deployment", "source": "..."}],
  "api_specs": [],
  "data_models": [],
  "migration": {},
  "risks": [],
  "review": [],
  "source_request_sha256": "..."
}
```

### 3.3 Benchmark schema

`taksh.arch_bench/1` — frozen before any training run.

```json
{
  "schema": "taksh.arch_bench/1",
  "suite_sha256": "...",
  "tasks": [],
  "rubric": {},
  "negative_controls": []
}
```

---

## 4. Runtime architecture

```text
requirements / repo context
        │
        ▼
   ArchitectContract
        │
        ▼
   Canonical renderer
        │
        ▼
   Gated model session (or labelled StaticModelDouble)
        │
        ▼
   Artifact parser
        │
        ├── validate.py   (schema, references, cycles, Mermaid/OpenAPI sanity)
        ├── review.py     (coupling, missing failure modes, security, operability)
        └── evidence.py   (MODEL_GENERATED / TOOL_OBSERVED / TOOL_VERIFIED / HUMAN_CONFIRMED)
        │
        ▼
   DesignArtifact bundle + qualification report
```

The coding engine consumes the bundle as context only. It does not receive write permission from this feature.

---

## 5. Validation rules

A valid bundle must satisfy:

- valid JSON/Markdown structure;
- all artifacts carry a request hash;
- ADR ids are unique and referenced;
- no dangling component references;
- dependency graph has no unexplained cycles;
- Mermaid diagrams parse at the structural level or are rejected;
- OpenAPI/AsyncAPI fragments contain required keys;
- security and operability risks are non-empty for production-oriented proposals;
- every assumption is marked `assumed`;
- vague requirements produce open questions, not invented constraints;
- no claims of correctness unless validated by independent review.

---

## 6. Evaluation

### 6.1 Frozen suite

Task categories:

- greenfield service design;
- brownfield integration;
- migration;
- API and event contract design;
- data model and consistency trade-offs;
- reliability and operability;
- security and compliance;
- performance and capacity;
- cost and rollback plan;
- ambiguity handling and clarification questions.

### 6.2 Metrics

| Metric | Meaning |
|---|---|
| requirement coverage | every stated requirement maps to an artifact |
| consistency | references across diagrams/ADRs/contracts agree |
| feasibility | design is plausible under declared constraints |
| trade-off quality | alternatives and consequences are explicit |
| operability | observability, rollback, incident response are addressed |
| security | auth, secrets, data handling and abuse cases are addressed |
| negative controls | vague prompts yield targeted questions |
| human rubric score | independent review on fixed rubric |

### 6.3 Qualification axes

Reports must separate:

1. engineering qualification;
2. model availability;
3. actual capability;
4. language qualification.

No combined "pass" claim is allowed.

---

## 7. Implementation phases

| Phase | Deliverable | Gate |
|---|---|---|
| P0 | This design, contracts and evaluation specification | owner architecture review |
| P1 | `taksh/architect/{contract,artifacts,render,validate,review}.py` | deterministic unit tests pass |
| P2 | `evaluation/architect/*`, frozen ARCH-BENCH | benchmark frozen before training |
| P3 | `taksh/architect/orchestrator.py`, `serving/architect_server.py`, domain routing | engineering-qualified, doubles labelled |
| P4 | governed architecture corpus | legal + owner authorization |
| P5 | SFT/DPO specialist checkpoint | authorized training and reproducibility |
| P6 | frozen benchmark qualification | owner release decision |

P0–P3 are CPU-only and can proceed without training authorization. P4 onward is owner-gated.

---

## 8. Data strategy

Candidate sources after authorization:

- architecture decision records with permissive licenses;
- open design docs and RFCs;
- OpenAPI/AsyncAPI corpora;
- architecture patterns and anti-patterns;
- migration reports;
- repo-grounded design tasks.

Requirements:

- explicit license allowlist;
- per-source provenance and hash;
- contamination scan against frozen benchmarks;
- deduplication;
- no copyrighted private material without authorization;
- no external data at this stage.

---

## 9. Safety and governance

- No repository mutation.
- No generated code execution in the design path.
- No production secrets in prompts.
- No tool calls outside explicitly approved read-only context sources.
- Human review remains mandatory for external or approval candidates.
- All outputs are labelled model-generated until independently validated.
- Failure is reported as blocked/unavailable, never silently substituted.

---

## 10. Deferrals

Deferred:

- implementation generation from design artifacts;
- autonomous design-to-code execution;
- external architecture retrieval;
- Karkain-specific tooling until KCC works;
- any claim that Design Architect is production-ready;
- any benchmark result before the suite is frozen.
