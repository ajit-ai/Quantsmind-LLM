# Taksh S2 Neural Failure Diagnosis — v1.5.0 Week 1

- **Date:** 2026-11-07T08:47:38.771010+00:00
- **Checkpoint:** `F:\Codes\Git\Quantsmind-LLM\artifacts\taksh-s2-experimental\taksh-s2.pt`
- **Checkpoint SHA-256:** `02cc86eb64022a454c0c7a795f18507cebbde709f295c65cd990d5e6f837f8ac`
- **Tokenizer:** `F:\Codes\Git\Quantsmind-LLM\tokenizer\qm_code_bpe_v0.json`
- **Tokenizer SHA-256:** `90fd1c8b968edf9c7eb841ef41e89dd7cea898239a589c1e5eb46ae233c6c988`
- **Model config:** d_model=256, n_layers=8, n_heads=8, n_kv_heads=4, ffn_dim=1024, vocab_size=5822, max_seq_len=512, tie_embeddings=True
- **Device:** cpu

## Summary

- **Total prompts:** 20
- **Passed (syntax valid + expected tail present):** 1/20 (5.0%)
- **Fail rate:** 95.0%
- **Syntax valid rate:** 60.0%

### Failure Category Counts

- **no_return_statement:** 11
- **syntax_error:** 7
- **other:** 1
- **empty_or_trivial:** 1

## Per-Prompt Results

| ID | Status | Syntax | Expected Tail | Category | Details |
|---|---|---|---|---|---|
| p01_add | FAIL | True | False | no_return_statement | no return statement in generation |
| p02_mul | FAIL | True | False | no_return_statement | no return statement in generation |
| p03_is_even | FAIL | False | False | syntax_error | full prompt+generation does not parse as Python; SyntaxError: invalid syntax at line 2 |
| p04_max | FAIL | True | False | no_return_statement | no return statement in generation |
| p05_factorial | FAIL | False | False | syntax_error | full prompt+generation does not parse as Python; SyntaxError: invalid syntax at line 5 |
| p06_reverse | FAIL | True | False | no_return_statement | no return statement in generation |
| p07_sum_list | PASS | True | True | other | — |
| p08_filter_pos | FAIL | False | False | syntax_error | full prompt+generation does not parse as Python; SyntaxError: invalid syntax at line 5 |
| p09_len | FAIL | True | False | no_return_statement | no return statement in generation |
| p10_is_pal | FAIL | True | False | no_return_statement | no return statement in generation |
| p11_count_vowels | FAIL | True | False | no_return_statement | no return statement in generation |
| p12_abs_val | FAIL | True | False | no_return_statement | no return statement in generation |
| p13_caps | FAIL | False | False | syntax_error | full prompt+generation does not parse as Python; SyntaxError: invalid syntax at line 2 |
| p14_merge | FAIL | False | False | syntax_error | full prompt+generation does not parse as Python; SyntaxError: invalid syntax at line 7 |
| p15_flatten | FAIL | False | False | syntax_error | full prompt+generation does not parse as Python; SyntaxError: invalid syntax at line 5 |
| p16_dedup | FAIL | True | False | no_return_statement | no return statement in generation |
| p17_c2f | FAIL | False | False | syntax_error | full prompt+generation does not parse as Python; SyntaxError: invalid syntax at line 2 |
| p18_prime | FAIL | True | False | no_return_statement | no return statement in generation |
| p19_sort | FAIL | False | False | empty_or_trivial | generation is empty or < 2 chars |
| p20_mod | FAIL | True | False | no_return_statement | no return statement in generation |

## Generated Outputs (Raw)

### p01_add
```python
def add(a, b):
    return left

```

### p02_mul
```python
def multiply(x, y):
    return left * right

```

### p03_is_even
```python
def is_even(n):
    return left left
```

### p04_max
```python
def maximum(a, b):
    if a > b:
        return a
    else:
        return right

```

### p05_factorial
```python
def factorial(n):
    if n <= 1:
        return 1
    else:
        return return return return return return return
```

### p06_reverse
```python
def reverse_string(s):
    return left - right

```

### p07_sum_list
```python
def sum_list(lst):
    total = 0
    for x in lst:
        total += x
    return total

```

### p08_filter_pos
```python
def filter_positive(nums):
    result = []
    for n in nums:
        if n > 0:
            result.lower    return left - right

```

### p09_len
```python
def string_length(s):
    return leftfunctionrect
```

### p10_is_pal
```python
def is_palindrome(s):
    return left - right

```

### p11_count_vowels
```python
def count_vowels(s):
    count = 0
    for ch in s:
        if ch in 'aeiou':
            count += 1
    return left

```

### p12_abs_val
```python
def absolute(x):
    if x < 0:
        return left

```

### p13_caps
```python
def capitalize_words(s):
    return left left - right

```

### p14_merge
```python
def merge_dicts(d1, d2):
    result = {}
    for k, v in d1.items():
        result[k] = v
    for k, v in d2.items():
        result[k] = v
    return left > > >lower values:
    return left > right:

```

### p15_flatten
```python
def flatten_list(nested):
    flat = []
    for sub in nested:
        for item in sub:
            flat.lower    return left

```

### p16_dedup
```python
def remove_duplicates(items):
    return left - right

```

### p17_c2f
```python
def celsius_to_fahrenheit(c):
    return left left *left in values:
    return left
```

### p18_prime
```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return left - right

```

### p19_sort
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] =       
```

### p20_mod
```python
def mod(a, b):
    return left - right

```
