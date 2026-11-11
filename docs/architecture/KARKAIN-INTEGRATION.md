# Karkain Integration — TAKSH × Karkain (L0.6.0-R1)

**Status:** integration contract specification (L0.6.0-R1). **No Karkain capability is claimed.** The local Karkain toolchain is partially broken on this host (evidence below); `karkain` CLI info-commands work, but every compilation-pipeline command exits with code 3 (ExitCompile) producing **zero diagnostics**, reproducible also with a freshly built reference compiler.

---

## 1. Karkain identity (preserved, not redesigned)

| Aspect | Value |
|---|---|
| Language id | `karkain` |
| Source files | `.kark` |
| Specification | `SPEC.md` **v1.1.0** — extracted from the reference compiler (`pkg/lexer`, `pkg/parser`, `pkg/sema`, `pkg/codegen`, `pkg/ir/ssa`); single source of truth |
| Compiler | `cmd/karkain` — Go reference compiler; engine name `kcc`; phase-based development |
| Compiler version pinned | **v1.1.0** (windows/amd64, "Stable Build"; branch `feature/v1.2.0-native-independence`; commit `e612ede`) |
| KIR | Karkain Intermediate Representation, **version v1**, SSA-based |
| Runtime | native executable emission (x86_64-windows native; C-free PE backend in-tree) |
| Standard library | `stdlib/`: async, collections, core, crypto, db, encoding, generics, gpu, http, io, math, net, numerics, string, system, testing |
| Self-hosting | `src/compiler/*.kark` — phase-gated, conforms to SPEC |

Karkain is TAKSH's **54th** first-class language. It does not replace any of the 53. It does not relax the 16-language L1.0 target. It does not exempt anyone from licensing review for training-corpus use.

## 2. KCC and KIR are the authoritative oracles — wherever available

`KCC` (the Karkain compiler) is the build/validate authority for:
- syntax validity (Error codes K000–K999 family);
- type/ownership/borrow checking;
- compilation success/failure;
- diagnostic truth (it — never TAKSH — decides what is a compile error).

`KIR v1` is the analysis authority for:
- structured IR inspection;
- semantics preservation;
- optimization behaviour (SSA form, phi nodes, dominance).

TAKSH must never invent Karkain syntax, semantics, compiler features, standard-library capabilities or error codes. When KCC disagrees with TAKSH, **KCC wins**. This contract is absolute.

## 3. Integration contracts (8)

### Contract 1 — `.kark` source comprehension and generation
TAKSH may parse, analyze and generate `.kark` source using the shared tokenizer. Generated code must then be checked by KCC before any "success" verdict. No string is "valid Karkain" except by KCC/KIR verdict.

### Contract 2 — Karkain lexical, grammatical and semantic knowledge
TAKSH's knowledge base stores Karkain concepts as Karkain-specific entries with provenance:

- lexical: keywords, operators (`:=`, `::`, arrow `->`, pipe `|>`), literals, comments, attributes
- grammatical: declarations, generics, traits/interfaces, ownership constructs, control flow, modules
- semantic: ownership & borrowing, move semantics, lifetime-connected rules, generics constraint syntax, mutability rules

### Contract 3 — KCC compilation and diagnostic feedback
Every generation must round-trip through KCC where available; diagnostics map 1:1 to TAKSH diagnostic codes; no "fixed" code is shipped without KCC passing afterward. Model-only and compiler-assisted scores are reported separately.

### Contract 4 — KIR comprehension and analysis
TAKSH can consume `karkain kir <file>` output; `--verify` flag validates SSA structure (dominance, phi placement); differences between model reasoning and KIR structure are logged as model defects, never patched over.

### Contract 5 — Karkain debugging and repair
dbg contract (`karkain dbg`): live-step backtrace with Karkain-level frames; debug sessions recorded; fixes validated by recompile; no silent repair.

### Contract 6 — Karkain tests and execution verification
`*_test.kark` is the executable truth. `karkain test <path>` discovers and runs tests. Model claims about "this Karkain program works" require a green `test` run (or a KCC compile + explicit exit-code-0) behind them.

### Contract 7 — Karkain repository-level engineering
Multi-file Karkain projects are qualified by: KCC build success, KCC test success, and repository-level diffs (imports, modules, test files, build files). Repository engineering is a separate qualification track from single-file fluency.

### Contract 8 — Cross-language translation involving Karkain
Karkain ↔ X translation must satisfy: (1) KCC accepts X→Karkain output; (2) round-trip Karkain→X→Karkain preserves semantics per KCC verdict; (3) regression on untouched Karkain features is zero; (4) reported only with toolchain evidence.

## 4. TAKSH-KARK-BENCH schema (contract, not results)

See `reports/l060r1_kark_bench_schema.json` (schema v1.0.0, validated by
`evaluation/l060r1_kark_bench_schema.py`). Ten categories: syntax, semantics,
compilation, KIR, debugging, algorithms, standard-library usage, translation,
repository engineering, adaptation to new Karkain features.

**Schema pins every result to a compiler + spec version.** Compiler-assisted
and model-only scores are always reported separately. No task is instantiated
in this milestone (see §6, actual toolchain state).

## 5. Local toolchain state (measured, this host — 2026-11-11)

| Check | Result |
|---|---|
| `karkain --version` | `Karkain Compiler v1.1.0 (windows/amd64, Stable Build)` — exit 0 |
| `karkain config` / `target` / `explain` | exit 0 |
| `kcc.exe` (in-tree Go reference binary) | Windows error **0xC0000135** — a required DLL is missing; cannot start |
| `karkain run/build/check/kir/lint/test/format` | all exit **3 (ExitCompile)** with **zero output** |
| Fresh `go build -o <temp> ./cmd/karkain` (Go 1.27.0) | build succeeds; runtime behaviour **identical** (silent exit 3) |
| `app.kark` golden sample | not accepted on this host (tooling failure, not code defect) |
| `.kark` file copied to temp | same silent failure — not a CWD/path artifact |

**Consequence.** On this machine TAKSH can read and document Karkain; it cannot
compile or execute Karkain. Any Karkain performance claim for this host would be
fabricated — none is made. The unblocking action is a Windows environment with
the missing runtime dependency(ies) for KCC (root-caused in the Karkain
repository, out of scope for this milestone).

## 6. What this milestone authorized vs deliberately forbade

**Authorized:** reading the Karkain repository at `F:\Codes\Git\Karkain` to
document its actual state and record limitations. **Forbidden (all respected):**
no modification to Karkain; no build writes into it (fresh compiler built to a
temp directory); no commit/tag/release/push there; no claim of Karkain capability.

## 7. Owner decisions outstanding (for L0.6.1 or later)

1. Repair the Karkain toolchain root cause on this host (or provide a healthy host/runner).
2. Authorize construction of TAKSH-KARK-BENCH tasks once KCC works.
3. Authorize benchmark execution against Karkain reference samples (still counts as execution, not training).
4. Authorization for any Karkain-derived training-corpus material (currently **zero** Karkain code in TAKSH training data).
