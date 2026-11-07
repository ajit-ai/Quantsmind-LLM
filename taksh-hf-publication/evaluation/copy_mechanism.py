"""Deterministic copy mechanism (Taksh v1.2.0 Increment 13, WP1-WP3).

Separates *which identifiers may be copied* (explicit candidates from the
prompt signature) from *how the model generates* (existing prediction
head, untouched). The gate never substitutes an unrelated identifier:
ambiguous or low-confidence cases abstain (fall back to the raw model
output) instead of guessing.
"""

from __future__ import annotations

import ast
import re

_BUILTINS = frozenset({
    "len", "max", "min", "sum", "abs", "sorted", "range", "list", "dict",
    "set", "tuple", "str", "int", "float", "bool", "isinstance", "enumerate",
    "zip", "any", "all", "reversed", "round", "filter",
})

_KEYWORDS = frozenset({
    "def", "return", "if", "else", "for", "while", "in", "not", "and", "or",
    "True", "False", "None", "import", "from", "as", "with", "class",
    "lambda", "pass", "break", "continue", "try", "except", "finally",
    "raise", "yield", "global", "nonlocal", "assert", "del", "is",
})


class SourceIdentifier:
    def __init__(self, name: str, token_ids: list[int], origin: str = "signature"):
        self.name = name
        self.token_ids = list(token_ids)
        self.origin = origin
        self.valid = bool(name) and name not in _KEYWORDS and name not in _BUILTINS

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "token_ids": self.token_ids,
            "token_count": len(self.token_ids),
            "origin": self.origin,
            "valid": self.valid,
        }


def extract_signature(prompt: str) -> str | None:
    matches = list(re.finditer(r"def\s+[A-Za-z_]\w*\([^\n]*\):", prompt or ""))
    if not matches:
        return None
    return matches[-1].group(0)


def extract_source_identifiers(prompt: str, tokenizer) -> list[SourceIdentifier]:
    """Extract copyable source identifiers from the prompt signature.

    AST-based: only function argument names are candidates (never
    keywords, builtins, or literals). token_ids from the frozen
    tokenizer; token boundaries are verified to spell the name exactly.
    """
    signature = extract_signature(prompt)
    if signature is None:
        return []
    function = ast.parse(signature + "\n    pass").body[0]
    candidates = []
    for arg in function.args.args:
        encoded = tokenizer.encode(arg.arg, add_special_tokens=False)
        round_trip = tokenizer.decode(encoded.ids) == arg.arg
        valid = round_trip and arg.arg not in _KEYWORDS and arg.arg not in _BUILTINS
        candidate = SourceIdentifier(arg.arg, encoded.ids, origin="signature")
        candidate.valid = valid
        candidates.append(candidate)
    return candidates


def _indent(body: str) -> str:
    lines = [line.rstrip() for line in (body or "").splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    if lines[0] and not lines[0][0].isspace():
        lines = ["    " + line if line else line for line in lines]
    return "\n".join(lines)


def _parse(signature: str, body: str):
    try:
        return ast.parse(signature + "\n" + _indent(body)).body[0]
    except SyntaxError:
        return None


def analyse_body(signature: str, body: str) -> dict:
    """Used params, invented names (loads not in params/builtins/locals)."""
    function = _parse(signature, body)
    if function is None:
        return {"parses": False, "used": [], "invented": [], "locals": []}
    param_names = [arg.arg for arg in function.args.args]
    assigned = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            assigned.add(node.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
        elif isinstance(node, ast.arg):
            assigned.add(node.arg)
    loaded = [
        node.id for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    ]
    used = [name for name in loaded if name in param_names]
    invented = sorted({name for name in loaded if name not in param_names and name not in assigned and name not in _BUILTINS})
    return {"parses": True, "used": used, "invented": invented, "locals": sorted(assigned - set(param_names))}


def select_substitutions(
    signature: str,
    body: str,
    candidates: list[SourceIdentifier],
    instruction: str = "",
) -> dict:
    """Map invented identifiers to source candidates deterministically.

    Policy: unused source params are matched to invented names in order of
    first appearance. Relation hints (`first`/`leading`/`second`/
    `trailing`) override positional choice. Any count mismatch or
    ambiguity returns ``action="abstain"`` — never a guess.
    """
    names = [c.name for c in candidates if c.valid]
    analysis = analyse_body(signature, body)
    if not analysis["parses"]:
        return {"action": "abstain", "mapping": {}, "reason": "unparseable", "confidence": 0.0}
    params = [arg.strip() for arg in signature.split("(", 1)[1].split(")", 1)[0].split(",") if arg.strip()]
    used = set(analysis["used"])
    invented = analysis["invented"]
    unused = [name for name in params if name not in used]
    text = (instruction or "").lower()
    mapping: dict[str, str] = {}
    if invented and unused:
        if "first" in text or "leading" in text:
            target = unused[0]
            mapping = {invented[0]: target}
        elif "second" in text or "trailing" in text:
            target = unused[0]
            mapping = {invented[0]: target}
        elif len(invented) == len(unused):
            mapping = {source: target for source, target in zip(invented, unused)}
        else:
            return {
                "action": "abstain",
                "mapping": {},
                "reason": f"ambiguous: {len(invented)} invented vs {len(unused)} unused",
                "confidence": 0.0,
            }
    if mapping:
        return {"action": "substitute", "mapping": mapping, "reason": "positional-unique", "confidence": 1.0}
    if not invented:
        return {
            "action": "propagate" if analysis["locals"] else "copy",
            "mapping": {},
            "reason": "no invented identifiers",
            "confidence": 1.0,
        }
    return {"action": "generate", "mapping": {}, "reason": "no source available", "confidence": 0.0}


class CopyGate:
    """Configurable copy-versus-generate gate (WP3).

    ``enabled=False`` disables all substitution (model output passes
    through untouched). ``threshold`` is the minimum confidence for a
    copy action; below it the gate abstains to the raw model output.
    Every decision carries an audit record.
    """

    def __init__(self, enabled: bool = True, threshold: float = 0.5):
        self.enabled = enabled
        self.threshold = threshold
        self.audit: list[dict] = []

    def decide(self, instruction: str, signature: str, body: str, candidates) -> dict:
        if not self.enabled:
            decision = {"action": "generate", "mapping": {}, "reason": "gate disabled", "confidence": 1.0}
        else:
            decision = select_substitutions(signature, body, candidates, instruction)
            if decision["confidence"] < self.threshold:
                decision = {**decision, "action": "abstain", "confidence": decision["confidence"]}
        decision["enabled"] = self.enabled
        decision["threshold"] = self.threshold
        self.audit.append(decision)
        return decision

    def apply(self, body: str, decision: dict) -> str:
        if decision["action"] not in ("substitute",):
            return body
        mapping = decision["mapping"]
        out = body
        # Longest names first so partial overlaps cannot clash.
        for invented, source in sorted(mapping.items(), key=lambda item: -len(item[0])):
            out = re.sub(rf"\b{re.escape(invented)}\b", source, out)
        return out


def copy_generate(gate: CopyGate, instruction: str, signature: str, body: str, tokenizer) -> dict:
    """Run extraction + gate + application over one model completion."""
    candidates = extract_source_identifiers(instruction + "\n" + signature, tokenizer)
    decision = gate.decide(instruction, signature, body, candidates)
    rewritten = gate.apply(body, decision)
    return {
        "completion": body,
        "candidates": [c.as_dict() for c in candidates],
        "decision": {k: v for k, v in decision.items()},
        "rewritten": rewritten,
        "changed": rewritten != body,
    }
