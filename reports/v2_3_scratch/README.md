# v2.3 scratch and interrupted implementation

This directory preserves the interrupted v2.3 generation-diagnostics work without shipping it in the Taksh package.

- `_append*.py`, `_w*.py`, `_write_detection.py`: one-off code-generation scripts used to assemble the modules.
- `INTERUPTED_V2.3_IMPLEMENTATION_REPORT.md`: the report written during the interrupted run. It contains inaccurate examples and must not be cited as verified evidence.
- `taksh/v2_3_*.py` were moved to import-safe names in `taksh/`.
- `tests/test_v2_3_generation_diagnostics.py` contains the renamed regression tests.
- `scripts/v2_3_verify_diagnostics.py` preserves the manual verification script.

The v2.3 diagnostics are experimental code, not a released Taksh capability. No model-capability or release claim is authorized by their presence. Simple one-statement functions such as `def f(x): return x` are intentionally treated as non-degenerate by the current detector; degeneration detection is heuristic and cannot infer task correctness.
