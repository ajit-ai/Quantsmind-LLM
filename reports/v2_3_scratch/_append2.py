content = '''

def detect_premature_eos(completion, expected_len_tokens=None):
    if not completion.strip():
        return {"premature_eos": False, "incomplete_function": False, "reason": "", "open_parens": 0, "open_braces": 0, "open_brackets": 0}
    result = {"premature_eos": False, "incomplete_function": False, "reason": "", "open_parens": 0, "open_braces": 0, "open_brackets": 0}
    result["open_parens"] = completion.count("(") - completion.count(")")
    result["open_braces"] = completion.count("{") - completion.count("}")
    result["open_brackets"] = completion.count("[") - completion.count("]")
    stripped = completion.rstrip()
    if not stripped:
        result["premature_eos"] = True
        result["reason"] = "empty completion"
        return result
    last_char = stripped[-1]
    if last_char in "([{":
        result["premature_eos"] = True
        result["reason"] = "completion ends with unclosed delimiter: %s" % last_char
        return result
    if expected_len_tokens is not None:
        actual_tokens = len(stripped.split())
        if actual_tokens < expected_len_tokens * 0.5 and expected_len_tokens > 0:
            result["premature_eos"] = True
            result["reason"] = "early termination: %d tokens vs expected ~%d" % (actual_tokens, expected_len_tokens)
    if result["open_parens"] != 0 or result["open_braces"] != 0 or result["open_brackets"] != 0:
        result["premature_eos"] = True
        parts = []
        if result["open_parens"] > 0:
            parts.append("%d open parens" % result["open_parens"])
        if result["open_braces"] > 0:
            parts.append("%d open braces" % result["open_braces"])
        if result["open_brackets"] > 0:
            parts.append("%d open brackets" % result["open_brackets"])
        result["reason"] = "; ".join(parts) + " - likely incomplete"
    if not result["premature_eos"] and not result["incomplete_function"]:
        token_count = len(stripped.split())
        if token_count <= 2:
            result["premature_eos"] = True
            result["reason"] = "very short completion: %d tokens" % token_count
    return result


def detect_excessive_repetition(completion, max_repeat_ratio=0.3):
    if not completion.strip():
        return {"excessive_repetition": False, "truncated": False, "repeat_ratio": 0.0, "unique_ratio": 0.0, "reason": ""}
    result = {"excessive_repetition": False, "truncated": False, "repeat_ratio": 0.0, "unique_ratio": 0.0, "reason": ""}
    tokens = completion.split()
    if len(tokens) < 3:
        result["truncated"] = True
        result["reason"] = "too short: %d tokens" % len(tokens)
        return result
    seen = set()
    repeat_count = 0
    for token in tokens:
        if token in seen:
            repeat_count += 1
        else:
            seen.add(token)
    unique_count = len(seen)
    repeat_ratio = repeat_count / len(tokens) if tokens else 0.0
    unique_ratio = unique_count / len(tokens) if tokens else 0.0
    result["repeat_ratio"] = round(repeat_ratio, 4)
    result["unique_ratio"] = round(unique_ratio, 4)
    if repeat_ratio >= max_repeat_ratio:
        result["excessive_repetition"] = True
        from collections import Counter
        token_freq = Counter(tokens)
        repeat_types = ["%s (%dx)" % (t, c) for t, c in token_freq.most_common(5) if c > 1]
        result["reason"] = "excessive repetition: %.1f%% of tokens are repeats (%s)" % (repeat_ratio * 100, ", ".join(repeat_types) if repeat_types else "unknown")
    stripped = completion.rstrip()
    if stripped:
        last_token = stripped.split()[-1] if stripped.split() else ""
        if "(" in last_token and ")" not in last_token:
            result["truncated"] = True
            if not result["excessive_repetition"]:
                result["reason"] = "output truncated mid-function-call"
            else:
                result["reason"] += "; output truncated mid-function-call"
    return result

'''

with open(r"F:\Codes\Git\Quantsmind-LLM\taksh\v2.3_detection.py", "a", encoding="utf-8") as f:
    f.write(content)
print("Append 2a done")