# This script writes the detection module
import textwrap

# Part 1: imports and dataclasses
part1 = '''"""Taksh v2.3.0 — Generation degeneration detection.

Implements diagnostics for generation-side failures that ordinary training
loss and syntax-validity metrics cannot detect.

Work Package 1 of Taksh v2.3.0.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class DetectionWarning:
    """A single detection warning with severity and evidence."""

    type: str
    severity: str  # "warning" or "failure"
    message: str
    evidence: Optional[Dict[str, Any]] = None
    location: Optional[str] = None


@dataclass
class DegenerationResult:
    """Complete degeneration detection result for one generation."""

    generation_id: str
    warnings: List[DetectionWarning]
    classification: str  # "pass", "warning", "fail"
    total_tokens: int
    unique_tokens: int
    repeat_rate: float
    repeated_token_sequences: List[Tuple[str, int]]
    repeated_operator_sequences: List[Dict[str, Any]]
    repeated_ast_subtrees: List[Dict[str, Any]]
    premature_eos: bool
    excessive_repetition: bool
    degenerate_semantics: bool


'''

with open(r"F:\Codes\Git\Quantsmind-LLM\taksh\v2.3_detection.py", "w", encoding="utf-8") as f:
    f.write(part1)
print("Part 1 written")