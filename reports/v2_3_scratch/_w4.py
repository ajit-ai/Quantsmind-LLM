content = 'def test_composition_analysis():\n    result = analyse_composition("def add_mul(a, b):", "    return a + b * 2")\n    assert result["parses"]\n    assert "add" in result["operations"]\n    assert "multiply" in result["operations"]\n\n\ndef test_composition_depth():\n    result = analyse_composition("def f(x):", "    if x > 0: return x")\n    assert result["parses"]\n    assert result["depth"] >= 1\n\n\ndef test_composition_novel_ops():\n    result = analyse_composition("def f(x):", "    return foo(x)")\n    assert "foo" in result["novel_operations"]\n\n\n'

with open(r"F:\Codes\Git\Quantsmind-LLM\taksh\v2.3_tests.py", "a", encoding="utf-8") as f:
    f.write(content)
print("WP3 written")