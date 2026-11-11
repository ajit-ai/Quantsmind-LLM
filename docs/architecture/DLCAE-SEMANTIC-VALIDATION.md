# DLCAE — Semantic and Execution Validation Foundation

**Status:** implemented (L1.0.0-ENG2, 2026-11-11). Engineering qualification only —
**no capability claim of any kind is authorized here**. Training handoff DISABLED
(A1.1-C / A2-E). The canonical 111M checkpoint remains MISSING.

This document is the developer reference for the ENG2 semantic layer. The architecture
remains `docs/architecture/DLCAE-ARCHITECTURE.md`; this file documents what is now
executable on top of it.

## 1. Pipeline position

```
Source
  → Incremental Comprehension (L1.0.0-ENG)
  → Symbol Analysis        (taksh/dlcae/semantic/symbols.py)
  → Type Analysis          (taksh/dlcae/semantic/typesys.py)
  → Dependency Analysis    (taksh/dlcae/semantic/dependencies.py)
  → Validation             (tool_assisted | model_only; fail-closed)
  → Knowledge Formation    (OBSERVED → VERIFIED only via Validation PASS)
  → Feedback               (records, correlation, repair candidates)
```

Nothing skips comprehension; nothing promotes past a missing toolchain; nothing
auto-modifies source.

## 2. Semantic contract (`taksh/dlcae/semantic/contract.py`)

`taksh.dlcae.semantic_contract/1` (contract version 1):

- statuses (exactly six): `PASS`, `FAIL`, `PARTIAL`, `BLOCKED`, `UNVERIFIED`, `NOT_APPLICABLE`
- `validate_result()` / `validate_diagnostic()` refuse: unknown statuses, missing
  toolchain with status PASS (missing toolchain never PASS), bad categories
  (`DIAGNOSTIC_CATEGORIES`), bad severities (`SEVERITIES`), locations below line 1.
- `build_result()` emits the canonical envelope: schema, contract version, language id +
  registry version, source SHA-256, module/file identity, diagnostics, symbols,
  references, type relationships, dependency relationships, comprehension summary,
  toolchain identity, provenance (source hash + producing modular run id), unsupported
  constructs.
- `merge_status()`: FAIL anywhere ⇒ FAIL; BLOCKED with nothing qualified ⇒ BLOCKED;
  BLOCKED mixed with PASS/PARTIAL ⇒ PARTIAL; PARTIAL ⇒ PARTIAL; else PASS.
- Release rule: a SemanticResult with `contract_problems == []` is structurally valid —
  still never capability evidence.

## 3. Symbol analysis

Language-neutral record format; language-specific rules inside adapters.

- `PythonSymbolAdapter` — real `ast`-based analysis: declarations
  (function/class/variable/import/parameter), qualified scope paths
  (`C.m`, `f.x`), scope boundaries, shadowing (inner binding over outer), duplicates
  (function/class re-definitions), unresolved identifiers (nearest-scope resolution →
  builtins → UNRESOLVED), import lists, module boundaries, exported flags (leading
  underscore excluded), source locations, provenance. Confidence recorded per
  resolution (0.9 same-scope / 0.75 outer-scope / 1.0 builtins).
- `GenericSymbolAdapter` — heuristic fallback; everything INFERRED/UNVERIFIED.
- `KarkainSymbolAdapter` — always **BLOCKED** (KCC 0xC0000135); no symbol table
  is invented from spec reading; bypassing KCC is prohibited.

## 4. Type analysis

Language-neutral, discipline-preserving records — no universal type system is imposed:

- `PythonTypeAdapter` — `ast` annotation extraction: signatures, parameters, returns,
  annotated variables, generics (`Generic[T]` → `type_parameter` records), all tagged
  `discipline: "gradual"`. Absent annotations are recorded as **explicit unknown**
  (`explicit_unknown: true`), never inferred silently.
- `types_compatible()` returns observations (`IDENTICAL`, `COMPATIBLE_WIDENING`,
  `INCOMPATIBLE_OR_UNKNOWN`, `UNVERIFIED`) — never verdicts. Promotion happens only
  through the Validation Engine.
- `KarkainTypeAdapter` — **BLOCKED**; discipline `static` recorded; no checker invented.

## 5. Dependency analysis

Local import graphs only — no package resolution, no remote registries, nothing downloaded:

- stdlib/local/third-party classification (`sys.stdlib_module_names`,
  `importlib.util.find_spec` for presence-only metadata; `importlib.metadata.version`
  for installed versions), relative imports resolved against the provided module map,
  cycles detected via DFS, missing dependencies detected and reported, impact analysis
  via reverse dependencies (`impact_analysis`).
- SHA-pinned: `hash_of_sources()` covers the whole source set.

## 6. Toolchain adapters (`taksh/dlcae/semantic/tools.py`)

Common interface: discovery, version inspection, availability, syntax validation,
semantic/static analysis, compilation/interpretation where safe, bounded execution,
temp-isolated workspace (`Sandbox`), timeouts, cancellation tokens, environment
fingerprint (`SHA-256` of tool identity), output bounded at 64 KiB.

Discovered on this host (2026-11-11):

| Language | Tool | Version | Level |
|---|---|---|---|
| Python | python + py_compile + mypy + ruff | 3.14.7 / 2.3.0 / 0.16.2 | syntax + semantic + lint |
| JavaScript | node --check | v24.18.1 | syntax |
| Go | gofmt parse oracle | go1.27.0 | syntax |
| Java | javac (JDK) | 21.0.2 | syntax/compiler diagnostics |
| Bash | bash -n | 5.2.37 | syntax |
| C#/.NET | dotnet (version only) | 10.0.400 | discovered; NOT_APPLICABLE for analysis |

Missing (TOOLCHAIN_UNAVAILABLE): TypeScript (tsc), Rust, MSVC/gcc, SQLite.
A missing toolchain yields BLOCKED verdicts; nothing reports PASS.

## 7. Feedback and failure localization

- `FeedbackRecord` — source hash, language, adapter + version, toolchain + version,
  diagnostic category, locations, execution context, reproducibility metadata,
  evidence classification (`TOOL_VERIFIED | MODEL_INFERRED | HEURISTIC | UNVERIFIED`).
- `correlate()` — deterministic primary/cascade classification (syntax > missing dep >
  compiler > unresolved > type > runtime > tests); groups by file:line; uncertainty
  preserved when competing peers exist; verdict `ROOT_CAUSE_IDENTIFIED` or
  `ROOT_CAUSE_UNCERTAIN`, never a guess.
- `repair_candidates()` — suggestions only: `source_mutation: false`, `verified: false`,
  `origin: suggestion-not-verified-fix`. No code path modifies source.

## 8. Knowledge promotion and invalidation

- Symbol facts are created as `OBSERVED` knowledge records; they become `VERIFIED`
  **only** when the validation stage reports PASS (tool-assisted run) — via
  `KnowledgeFormationEngine.verify(validation_verdict="PASS")`.
- `SourceEvidenceLedger` binds every VERIFIED binding to a source SHA-256; a changed
  source invalidates all bindings to the old hash (`INVALIDATED`, explicit reason;
  nothing silently retained, never re-promoted automatically).

## 9. Interrupted work and resume

`SemanticAnalysisEngine.analyze(..., stop_after=<stage>)` persists the interrupted
ledger (state_store when provided) and returns a `resume_token`.
`analyze(..., resume_token=…)` continues from completed stages. A token presented
with different source text is **refused** (fail-closed). `verify_source()`/`invalidate`
is the runtime path for hash-driven invalidation.

## 10. Experiments

`taksh/dlcae/semantic/metrics.py` reports synthetic ground-truth metrics:
symbol precision/recall, unresolved-identifier detection, shadowing, duplicates,
type-relation kinds, dependency cycles/missing detection, reproducibility fingerprints,
memory traces and the compatibility matrix. Arms A/B/C are **unchanged**; semantic
metrics extend measurement, not conclusions;
comparison verdict remains `NOT_ESTABLISHED`; **no comparative winner exists**.

## 11. What this layer does NOT establish

No learning; no model improvement; no language qualification; Karkain remains
DISCOVERED with KARK-BENCH at zero tasks; frozen Python benchmark 0/30; v180_sft
unusable (geometry); canonical 111M checkpoint missing; A1.1-C/A2-E unchanged.
