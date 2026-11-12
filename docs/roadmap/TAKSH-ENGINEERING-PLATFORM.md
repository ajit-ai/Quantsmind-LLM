# TAKSH Integrated Coding Engineering Platform

**Status:** Source implementation and focused tests pass. This is engineering-platform
evidence, not native model capability.

**Verification:** `tests/test_engineering_platform.py` — 13 passed, 1 skipped.

---

## 1. Architecture

The platform lives in `taksh/engineering.py` and unifies:

- Project understanding (`taksh/project_understanding.py`)
- Specialist routing (`taksh/specialists.py`)
- Coding validation (`taksh/coding_validation.py`)
- Task orchestration (`taksh/coding_engine.py`)
- Safe patch application and rollback (new `PatchTransaction`)
- Test discovery and execution (new)
- Bounded repair (new)

DLCAE components are not present in this checkout, so integration was performed
against the existing TAKSH components. Their responsibilities are preserved rather
than duplicated:

| Responsibility | Owner |
|---|---|
| Project comprehension | `taksh/project_understanding.py` |
| Specialist selection | `taksh/specialists.py` |
| Change validation | `taksh/coding_validation.py` |
| Coding orchestration | `taksh/coding_engine.py` |
| Safe multi-file application | `taksh/engineering.PatchTransaction` |
| Test execution | `taksh.engineering.run_tests` |
| Repair loop | `taksh.engineering.run_repair_loop` |

## 2. Implemented workflows

- Project inspection: recursive discovery, symbol indexing, dependency graph,
  test discovery.
- Task planning: task model, specialist routing, provider-based change generation.
- Multi-file change plans: create, modify, delete operations with reasons and
  dependencies.
- Safe patch application: workspace-root enforcement, path-traversal protection,
  symlink-escape protection, conflict detection, snapshots and rollback.
- Validation: syntax, imports (including project-local modules), semantics.
- Test discovery and execution: pytest with bounded timeout and captured output.
- Bounded repair: configurable iteration limit, traceable repair history.
- Provider integration: native provider fails clearly when unavailable;
  deterministic test provider is explicitly identified.
- CLI: `taksh engineer inspect|validate|test|plan|apply|run`.
- SDK: module-level functions and aliases.

## 3. Security model

- All paths are resolved and checked against the workspace root.
- `..`, absolute paths and symlink escapes are rejected.
- Patch application snapshots originals and rolls back on failure.
- Unapproved plans are not applied.
- Conflicting file modifications are detected before overwrite.
- Test execution uses a timeout and captures bounded output.
- No implicit network access or external tool execution.

## 4. End-to-end demonstration

`tests/test_engineering_platform.py::test_end_to_end_multi_file_engineering`
creates an isolated temporary Python project, uses a deterministic provider to
modify one file, create two files, applies the changes, validates syntax and
imports, runs targeted tests, and verifies:

- all intended files changed;
- unrelated files remained unchanged;
- validation passed;
- tests passed;
- no capability claim was made.

Additional tests cover invalid patches, conflicts, workspace escape, rollback,
provider-unavailable behaviour and bounded repair.

## 5. Known limitations

- Native TAKSH model capability remains `NOT_EVALUATED`.
- The deterministic provider is a test double, not model evidence.
- Semantic validation is lightweight and does not type-check.
- Test execution is local pytest only; no container isolation is implemented.
- No IDE/GUI integration is implemented.
- No claim of TAKSH 2.0.0 GA readiness is made.

## 6. Version impact

- `1.3.0` remains `PARTIAL`: coding intelligence platform improved, but
  capability is not evaluated.
- `1.4.0` remains `PARTIAL`: multi-file engineering workflow implemented, but
  not product-qualified.
- Package version remains `1.0.0`.
- No release authorization exists.
