import sys
sys.path.insert(0, r'F:/Codes/Git/Quantsmind-LLM')
import taksh.v2_3_detection as d
detect_degeneration = d.detect_degeneration

# Test all regression cases
tests = [
    ("def f(x): return x", "identity"),
    ("def f(x): return 5", "constant"),
    ("def f(x): return x + 1", "simple_add"),
    ("def f(x): return x * 2", "simple_mul"),
    ("def add(a, b): return a + b", "valid"),
    ("x = " + "foo " * 50, "repetition"),
    ("def f(x): return (1 +", "premature_eos"),
    ("def f(x): pass", "empty_body"),
]

print("Regression Test Results:")
print("=" * 60)
for code, label in tests:
    r = detect_degeneration(code, label)
    print(f"{label:15s}: class={r.classification:6s} deg={r.degenerate_semantics}")
    if r.warnings:
        for w in r.warnings:
            print(f"  warning: {w.type} {w.severity}: {w.message[:50]}")

print()
print("All identity/constant/simple functions should have classification=PASS")
print("Genuine degeneration should have classification=FAIL or WARNING")