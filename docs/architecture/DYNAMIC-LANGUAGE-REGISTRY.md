# Dynamic Language Registry — no fixed catalogue

**Status:** specification + validator (L0.6.0-R1). **53 languages preserved intact; Karkain is the 54th first-class target. No ceiling of any size is permitted anywhere.**

## 1. Why a dynamic registry

TAKSH's premise is that the catalogue must never be exhausted by design. Any
hard-coded language count (54 or otherwise) would bake today's inventory into
tomorrow's ceiling. The registry is therefore:

- open-ended, extension-based;
- validated by tools, not by convention;
- explicit about status (registration ≠ capability);
- pinned to provenance (license, version, toolchain, benchmarks);
- machine-readable; consumed by evaluation, not by marketing.

## 2. Schema and statuses

Entries live in `configs/foundation/l060r1_dynamic_language_registry.yaml`
(extension entries); the immutable base lives in
`configs/feasibility/l051_languages.yaml` (53 languages, 16 L1.0 targets —
absolutely untouched).

```yaml
entries:
  - id: karkain
    name: Karkain
    aliases: [karkain-lang]
    version_or_dialect: "SPEC v1.1.0; reference compiler v1.1.0"
    extensions: [.kark]
    tokenizer_metadata: "..."
    grammar_or_spec_reference: "..."
    semantic_metadata: "..."
    compiler_or_runtime_integration: "..."
    execution_config: "..."
    stdlib_knowledge_reference: "..."
    documentation_and_provenance: "..."
    license_and_source_authorization: "..."
    qualification_status: DISCOVERED
    benchmark_associations: [TAKSH-KARK-BENCH v1.0.0]
    adaptation_history: []
```

### Status model (exactly six)

| Status | Meaning | Capability? |
|---|---|---|
| REGISTERED | id + intent recorded | No |
| DISCOVERED | docs/source/toolchain found, provenance recorded | No |
| COMPREHENSION_IN_PROGRESS | approved knowledge + progressive comprehension under way | No |
| FUNCTIONALLY_QUALIFIED | passes the dynamic-language benchmark bar | **Yes** |
| ENGINEERING_QUALIFIED | passes repository-engineering qualification | **Yes** |
| BLOCKED | license / toolchain / spec / safety barrier | No |

Promotion requires measurement; demotion on regression is mandatory; BLOCKED is
terminal until an explicit owner unblocks it.

## 3. Tooling shipped with this milestone

- `evaluation/l060r1_dynamic_registry.py` — loads the 53-language base registry,
  merges extension entries (54 languages today), validates structure,
  ids, statuses, Karkain contract, and emits
  `reports/l060r1_dynamic_language_registry.json`.
- Validation rules enforced on every load:
  - base registry validates (53 langs, 16 L1.0 targets, supporting tech separate)
  - extension ids never collide with existing ids
  - every required field present; `adaptation_history` may be empty but must exist
  - qualification requires FUNCTIONALLY_QUALIFIED or ENGINEERING_QUALIFIED
  - no ceiling of any kind (structurally verified)

```
python -m evaluation.l060r1_dynamic_registry
```

## 4. Adding a new language (today)

1. Append an entry to `l060r1_dynamic_language_registry.yaml` with all `required_fields`.
2. Assign status REGISTERED (or DISCOVERED with real provenance).
3. Run the validator; fix reported issues.
4. Do NOT assign a capability status; the Evaluation Engine must measure first.

No foundation-model configuration changes are required per new language.

## 5. Deliverable evidence (this milestone)

- 53-language base preserved byte-identically (hash recorded).
- Karkain present as the 54th entry, status DISCOVERED.
- merged view: 54 languages; no ceiling; Karkain associations intact.
- Validator rejects collision, missing fields, bad statuses, Karkain contract violations.
