"""Code-quality checks for generated output (M4 §12, §16).

Four statuses are reported **separately**, never collapsed into one score:

``SYNTAX_VALID``
    the extracted code parses as Python
``TEST_PASS`` / ``TEST_FAIL``
    it parses and ran against predefined authored assertions
``NOT_TESTABLE``
    no test was defined for this prompt
``EXTRACTION_AMBIGUOUS``
    more than one plausible code block, so nothing was extracted

Extraction is explicit, not clever: a fenced block if the model produced one,
otherwise the raw text. When that yields several candidates the item is marked
ambiguous rather than guessed at, because silently picking the one that parses
turns an honest failure into a fake success.

Execution is isolated: authored assertions run in a subprocess with a timeout.
No dataset-supplied code is ever executed.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SYNTAX_VALID = "SYNTAX_VALID"
SYNTAX_INVALID = "SYNTAX_INVALID"
TEST_PASS = "TEST_PASS"
TEST_FAIL = "TEST_FAIL"
NOT_TESTABLE = "NOT_TESTABLE"
EXTRACTION_AMBIGUOUS = "EXTRACTION_AMBIGUOUS"
NO_CODE = "NO_CODE"

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# Authored here, not generated. Each states a total contract, so a failure is
# meaningful rather than a matter of taste.
SAFE_TESTS = {
    "reverse_string": (
        "assert reverse_string('abc') == 'cba'\n"
        "assert reverse_string('') == ''"
    ),
    "is_palindrome": (
        "assert is_palindrome('racecar') is True\n"
        "assert is_palindrome('ab') is False"
    ),
    "fizzbuzz_n": (
        "out = fizzbuzz_n(5)\n"
        "assert out[0] == '1'\nassert out[2] == 'Fizz'\n"
        "assert out[4] == 'Buzz'\n"
        "assert fizzbuzz_n(15)[-1] == 'FizzBuzz'"
    ),
    "factorial": (
        "assert factorial(0) == 1\nassert factorial(5) == 120\n"
        "assert factorial(10) == 3628800"
    ),
    "sum_list": "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0",
    "count_vowels": "assert count_vowels('aeiou') == 5\nassert count_vowels('') == 0",
    "flatten": "assert flatten([1, [2, [3]]]) == [1, 2, 3]\nassert flatten([]) == []",
    "second_largest": (
        "assert second_largest([1, 3, 2]) == 2\n"
        "assert second_largest([9, 4, 4]) == 4\n"
        "assert second_largest([7]) is None"
    ),
}


def _has_executable_content(tree) -> bool:
    """True when the parsed module does anything.

    A block of ``#`` comments and stray ``` ``` `` fragments parses as valid
    Python -- every line is a comment -- so a naive ``ast.parse`` check scores
    pure noise as SYNTAX_VALID. That is exactly the metric gaming this module
    exists to prevent, so a snippet counts as code only when it contains at
    least one statement that is not a bare docstring.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Assign, ast.AugAssign,
                             ast.AnnAssign, ast.Import, ast.ImportFrom,
                             ast.Return, ast.If, ast.For, ast.While, ast.Try,
                             ast.Expr)):
            if isinstance(node, ast.Expr):
                # A bare string is a docstring; a call or name is real code.
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    continue
            return True
    return False


def extract_code(text):
    """Return ``(code, status)``. Never guesses between candidates."""
    if not isinstance(text, str) or not text.strip():
        return None, NO_CODE
    blocks = [block.strip() for block in _FENCE_RE.findall(text)
              if block.strip()]
    if len(blocks) > 1:
        return None, EXTRACTION_AMBIGUOUS
    if len(blocks) == 1:
        # A fenced block is the model asserting "this is my code", so it is
        # returned even when it fails to parse. SYNTAX_INVALID then reports the
        # failure; folding it into NO_CODE would hide "tried, got it wrong"
        # behind "produced nothing".
        return blocks[0], "SINGLE_BLOCK"

    candidate = text.strip()
    try:
        tree = ast.parse(candidate)
    except (SyntaxError, ValueError):
        return None, NO_CODE
    if not _has_executable_content(tree):
        return None, NO_CODE
    return candidate, "BARE_CODE"


def syntax_status(code):
    if not code:
        return NO_CODE
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return SYNTAX_INVALID
    return SYNTAX_VALID if _has_executable_content(tree) else NO_CODE


def run_safe_test(code, test_source, timeout=15):
    """Run authored assertions in a separate process with a timeout."""
    program = f"{code}\n\n{test_source}\nprint('OK')\n"
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8")
    handle.write(program)
    handle.close()
    path = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, path], capture_output=True, text=True,
            timeout=timeout, cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    finally:
        Path(path).unlink(missing_ok=True)
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()
        return False, tail[-1] if tail else f"exit {completed.returncode}"
    return True, "passed"


def evaluate_generation(prompt_id, prompt, output, test_source=None):
    """Full check for one generation. Each dimension is reported separately."""
    code, extraction = extract_code(output)
    syntax = syntax_status(code)
    result = {
        "prompt_id": prompt_id,
        "extraction_status": extraction,
        "syntax_status": syntax,
        "test_status": NOT_TESTABLE,
        "test_detail": None,
        "instruction_compliance": None,
        "compliance_detail": None,
        "output": output,
        "extracted_code": code,
        "output_chars": len(output or ""),
    }

    if test_source:
        if syntax != SYNTAX_VALID:
            result["test_status"] = NOT_TESTABLE
            result["test_detail"] = "not run: code does not parse"
        else:
            passed, detail = run_safe_test(code, test_source)
            result["test_status"] = TEST_PASS if passed else TEST_FAIL
            result["test_detail"] = detail

    compliant, detail = check_format_compliance(prompt, output)
    result["instruction_compliance"] = compliant
    result["compliance_detail"] = detail
    return result


def check_format_compliance(prompt, output):
    """Deterministic format constraints, where the instruction states one.

    Returns ``(True, ...)`` when the prompt states no machine-checkable rule,
    which keeps "not applicable" from being scored as a pass or a failure.
    """
    text = (output or "").strip()
    lowered = (prompt or "").lower()
    blocks = [b.strip() for b in _FENCE_RE.findall(text or "") if b.strip()]

    # A prompt that *asks* for a fenced block requires one. Scoring fences as a
    # violation here would penalise exactly the output the instruction requested.
    if "fenced" in lowered and "code block" in lowered:
        ok = len(blocks) == 1
        return ok, f"{len(blocks)} fenced block(s), expected exactly 1"

    if "exactly the word" in lowered:
        expected = prompt.split("exactly the word", 1)[1]
        expected = expected.split("and nothing", 1)[0].strip().strip(".").strip()
        ok = text.strip(".").upper() == expected.upper()
        return ok, (f"matched {expected!r}" if ok
                    else f"expected {expected!r}, got {text!r}")

    if "exactly three" in lowered and "bullet list" in lowered:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        ok = len(lines) == 3 and all(ln.strip().startswith("-") for ln in lines)
        return ok, f"{len(lines)} non-empty line(s), all bullets={ok}"

    if "exactly four" in lowered and "numbered list" in lowered:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        ok = len(lines) == 4 and all(ln.strip()[:1].isdigit() for ln in lines)
        return ok, f"{len(lines)} non-empty line(s), numbered={ok}"

    if "bare number" in lowered:
        ok = text.replace("-", "").replace(".", "").isdigit()
        return ok, f"output {text!r} is {'not ' if not ok else ''}a bare number"

    if "only the json" in lowered:
        candidate = blocks[0] if blocks else text
        try:
            json.loads(candidate)
            return True, "parsed as JSON"
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return False, f"not JSON: {exc}"

    if "nothing else" in lowered:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return len(lines) <= 1, f"{len(lines)} non-empty line(s)"

    if "only yaml" in lowered or "no fences" in lowered or "output only" in lowered:
        ok = len(blocks) == 0
        return ok, "no markdown fences" if ok else "contained markdown fences"

    return True, "no machine-checkable constraint in this prompt"


def summarise(results):
    """Aggregate. Dimensions stay separate; no composite score is invented."""
    n = len(results) or 1
    testable = [r for r in results
                if r.get("test_status") not in (NOT_TESTABLE, None)]
    scored = [r for r in results if r.get("instruction_compliance") is not None]
    syntax = sum(1 for r in results if r["syntax_status"] == SYNTAX_VALID)
    test_pass = sum(1 for r in testable if r["test_status"] == TEST_PASS)
    compliant = sum(1 for r in scored if r.get("instruction_compliance"))
    return {
        "items": len(results),
        "syntax_valid": syntax,
        "syntax_valid_rate": round(syntax / n, 4),
        "testable_items": len(testable),
        "test_pass": test_pass,
        "test_pass_rate": round(test_pass / len(testable), 4) if testable else None,
        "instruction_compliance_scored": len(scored),
        "instruction_compliance_rate": (
            round(compliant / len(scored), 4) if scored else None
        ),
        "mean_output_chars": round(
            sum(r.get("output_chars", 0) for r in results) / n, 1
        ),
    }