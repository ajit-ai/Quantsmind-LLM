# TAKSH L0.5.1 — Multilingual Feasibility Report

**Decision: CONDITIONAL GO** — engineering implementation delivered and verified; capability is demonstrated nowhere (0/30 Python, 0/16 multilingual starter); authorization to train, buy data, or rent GPUs is NOT granted.

## 1. Preservation evidence (verified 2026-11-11)
- Branch `main` @ `c601a73716d5`
- `main` = `c601a73716d5`; `develop` = `8c553465216d`; working tree entries: 81
- Tokenizer: vocab 5,822 (unchanged); all specials intact (pad=0 bos=1 eos=2 unk=3)
- Frozen Python benchmark SHA-256 `854c7ca833c224eb…` == frozen; `taksh-hf-publication/` untouched; package `taksh==1.0.0`

## 2. Language registry (WP1)
- 53 languages registered and validated: exactly 16 L1.0 targets; 12 enterprise; 10 scientific; 10 systems; 5 emerging.
- Supporting technologies kept separate: HTML, CSS, SCSS, JSON, YAML, XML, TOML, GraphQL, Terraform HCL, Nix, CMake, Makefiles, Gradle DSL, PowerShell, Zsh, OpenAPI, Protocol Buffers, Verilog, SystemVerilog, VHDL — none counted as programming languages.

## 3. Tokenizer coverage audit (WP2)
- 73 samples (53 languages + 20 supporting); roundtrip exact: **73/73**
- tokens/char range 0.301..0.633
- No language excluded by tokenization; indentation single-token; byte-exact decode for every sample.
- Audit measures representation, not capability.

## 4. Multilingual dataset feasibility (WP3)
- L1.0 verdicts: qualified-in-repo: ['python']; conditional-on-legal-review: 15; deferred: 0; blocked: []
- Global: 1 qualified (Python), 28 conditional, 24 deferred, 0 blocked
- in-repo ground truth: 1,590,703 Python tokens (measured); **zero tokens for every other L1.0 language**
- Nothing downloaded, scraped, or purchased; BigCode/CodeSearchNet/PyPI candidates registered, all require legal review

## 5. Training distribution model (WP4)
- equal_token: blocked (15 of 16 languages have zero unique supply)
- proportional: degenerates to 100% Python today
- quality_weighted: collapses to Python (only verified-quality signal)
- capability_gap: uniform-by-default (no capability measured anywhere)
- family_balance: 100% scripting(python), 0% elsewhere
- equal-language epoch at the 4x reuse ceiling: ~5.97M unique tokens missing (15 languages x ~398K each)

## 6. Architecture comparison (WP5)
- A (one shared multilingual model): PREFERRED HYPOTHESIS — not proven; cheapest, deployment-simplest, matches verified stack
- B (shared + language adapters): fallback if A shows destructive interference; 41.8M params
- C (shared + family adapters): variant of B; 35.3M params
- D (separate models): rejected for now — 460.6M params, no shared capability, absurd at current scale

## 7. Multilingual benchmark (WP6)
- 64 starter tasks; 16 languages; families 1-4 instantiated; families 5-10 specified but NOT instantiated (disclosed)
- Frozen Python benchmark SHA-256 unchanged: True
- Proposed threshold fixed before any evaluation: >= 8/16 tasks per language; Python bar unchanged; never renegotiated after results

## 8. Toolchain inventory (WP7)
- available (12/16): bash, c, cpp, csharp, go, java, javascript, kotlin, python, rust, sql, typescript
- missing (4/16): php, ruby, swift, dart
- SQL via Python sqlite3 (3.50.4); Bash via Git Bash 5.2.37; C/C++ via MSVC 19.44.35229 (vsdevcmd required); TypeScript via Node type-stripping only; no tsc

## 9. Compute and storage (WP8)
- checkpoint storage: 28.8M→115 MB, A→443 MB, B→1.16 GB, C→4.07 GB; training working sets ~16 bytes/param
- CPU inference of the current model verified (L0.1); 289M fits 16 GB VRAM classes for training; 1B needs ≥ 24 GB
- equal-split across 16 languages: 99,419 tokens/language from the full qualified pool — nothing to train with

## 10. Per-language risk classification (WP10)
| Classification | Count | Languages |
|---|---|---|
| feasible-for-L1.0-investigation | 1 | python |
| conditional-on-data/legal-review | 15 | bash, c, cpp, csharp, go, groovy, java, javascript, kotlin, perl, rust, scala, sql, typescript, vbnet |
| conditional-on-toolchain+data/legal-review | 4 | dart, php, ruby, swift |
| deferred | 28 | ada, assembly, clojure, cobol, crystal, delphi, elixir, erlang, fortran, fsharp, gdscript, haskell, julia, lisp, lua, matlab, micropython, mojo, nim, objectivec, ocaml, plsql, prolog, r, solidity, tsql, vba, zig |
| technically-unsupported-by-current-evidence | 5 | gleam, hack, odin, pony, vlang |

## 11. Decision (WP10)

### CONDITIONAL GO
Reasons:
- a multilingual shared-model program is technically plausible (tokenizer is adequate for all 73 audited samples, no roundtrip loss)
- BUT the qualified in-repo multilingual code supply is exactly 1.59M Python tokens and ZERO for every other language
- and no multilingual benchmark-executed capability exists for any language (0/30 on Python; 0/16 on the multilingual starter)
- and 10 of 16 L1.0 toolchains are locally usable but 4 (php/ruby/swift/dart) are not, and typescript has no compiler
- and no architecture beyond the verified 28.8M baseline is justified by evidence today

Owner approvals required for L0.6:
- written authorization to acquire licensed multilingual data (named corpora; legal sign-off per corpus)
- written authorization for GPU compute budget (class and duration)
- selection of the initial L1.0 language set to actually execute (subset of the 16 recommended by toolchain+data)
- confirmation that the multilingual benchmark acceptance bar (>= 8/16 per language) is binding
- decision on execution of TypeScript via Node type-stripping vs installing tsc (authorization required)
- confirmation that L0.1 default checkpoint remains untouched

Would flip to GO:
- owner-authorized acquisition of >= 20B tokens of license-verified multilingual code with clean legal review
- >= 24 GB VRAM-class compute for a >= 100M-parameter model, or equivalent CPU-only schedule accepted (~28 days)
- a candidate passing >= 8/16 multilingual tasks AND beating the frozen Python bar on its own benchmark

Would flip to NO-GO:
- owner declines data acquisition or compute budget
- legal review rejects all candidate corpora for the selected languages
- an authorized L0.6 experiment scores 0/16 across all languages with a clean benchmark


*Generated by `python -m evaluation.l051_feasibility`. This document and its JSON twin do not authorize training, data acquisition, compute rental, or publication. The L0.1 default checkpoint remains unchanged.*
