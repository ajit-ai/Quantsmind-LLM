content = '"""Taksh v2.3.0 — Regression and reproducibility tests."""\n\nfrom __future__ import annotations\n\nimport sys\n\nfrom taksh.v2_3_detection import detect_degeneration\nfrom taksh.v2_3_parameter_binding import analyse_parameter_binding, test_renamed_params\nfrom taksh.v2_3_composition import analyse_composition\nfrom taksh.v2_3_holdout_mapping import build_mapping_case, evaluate_mapping_case\nfrom taksh.v2_3_canary import run_canary, compare_to_baseline\n\n\n'

with open(r"F:\Codes\Git\Quantsmind-LLM\taksh\v2.3_tests.py", "w", encoding="utf-8") as f:
    f.write(content)
print("header written")