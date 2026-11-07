import pytest

from evaluation.eval_code import HUMANEVAL_PROBLEMS, MBPP_PROBLEMS, generate_completion, run_humaneval


def test_humaneval_problems_structure():
    assert len(HUMANEVAL_PROBLEMS) >= 5
    for prob in HUMANEVAL_PROBLEMS:
        assert "task_id" in prob
        assert "prompt" in prob
        assert "entry_point" in prob
        assert "test" in prob


def test_mbpp_problems_structure():
    assert len(MBPP_PROBLEMS) >= 5
    for prob in MBPP_PROBLEMS:
        assert "task_id" in prob
        assert "prompt" in prob
        assert "entry_point" in prob
        assert "test" in prob


def test_humaneval_python_syntax():
    for prob in HUMANEVAL_PROBLEMS:
        prompt = prob["prompt"]
        assert "def " in prompt
        assert prompt.strip().endswith('"""')


def test_mbpp_prompts_are_clear():
    for prob in MBPP_PROBLEMS:
        prompt = prob["prompt"]
        assert len(prompt) > 10
        assert "function" in prompt.lower() or "write" in prompt.lower()
