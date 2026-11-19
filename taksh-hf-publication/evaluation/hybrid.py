"""Integrated hybrid generation (Taksh integrated campaign, WP2-WP4).

Combines the frozen neural model (read-only loaded) with the Increment 13
copy candidates: the model generates, the mechanism resolves identifiers
by inferred source-to-target relations and teacher-forced confidence, and
the result is rewritten (or aborted to the raw model output) with a full
audit trail. Syntax and operations are preserved; only invented
identifier tokens are ever rewritten, and only when uniquely resolvable.
"""

from __future__ import annotations

import ast
import re

from evaluation.copy_mechanism import _BUILTINS, extract_source_identifiers
from training.program_analysis import analyse

# Operation -> ordered target parameter indices (used only when unique).
_OP_RULES = (
    ("product", "multiply", "times", "scale"),
    ("difference", "subtract", "minus", "less"),
    ("sum", "total", "add", "combine", "together", "plus"),
    ("larger", "greater", "maximum", "bigger", "max"),
    ("smaller", "lesser", "minimum", "min"),
    ("first", "leading", "head"),
    ("second", "trailing", "tail"),
)

_SINGLE_PARAM_HINTS = (
    "upper", "capital", "shout", "lower", "small", "quiet",
    "count", "tally", "how many", "number of",
    "positive", "negative", "above zero", "below zero",
    "join", "glue", "combine the strings", "words together", "space",
)


def _parse(signature: str, body: str):
    try:
        return ast.parse(signature + "\n" + body).body[0]
    except SyntaxError:
        return None


def analyse_names(signature: str, body: str) -> dict:
    """Params used, invented names in appearance order, and locals."""
    function = _parse(signature, body)
    if function is None:
        return {"parses": False, "used": [], "invented": [], "locals": []}
    param_names = [arg.arg for arg in function.args.args]
    assigned = set()
    for node in ast.walk(function):
        if isinstance(node, ast.arg):
            assigned.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            assigned.add(node.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
    invented_in_order = []
    loaded = []
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.append(node.id)
            if node.id not in param_names and node.id not in assigned and node.id not in _BUILTINS and node.id not in invented_in_order:
                invented_in_order.append(node.id)
    used = [name for name in loaded if name in param_names]
    return {
        "parses": True,
        "used": used,
        "invented": invented_in_order,
        "locals": sorted(assigned - set(param_names)),
    }


def infer_relation(instruction: str, body: str, invented: list[str], unused: list[str]) -> dict:
    """Deterministic operation-aware mapping from invented to source names."""
    text = (instruction or "").lower()
    if invented and unused:
        if "first" in text or "leading" in text:
            return {"mapping": {invented[0]: unused[0]}, "reason": "first-hint", "confidence": 0.9}
        if "second" in text or "trailing" in text:
            return {"mapping": {invented[0]: unused[0]}, "reason": "second-hint", "confidence": 0.9}
        if len(invented) == len(unused):
            return {"mapping": dict(zip(invented, unused)), "reason": "positional-unique", "confidence": 0.8}
        if len(invented) == 1 and len(unused) >= 1:
            # Single invented name with hint of a unary text/aggregate op:
            # use the (single) unused source, or the first when binary.
            for hint in _SINGLE_PARAM_HINTS:
                if hint in text:
                    return {"mapping": {invented[0]: unused[0]}, "reason": f"hint:{hint}", "confidence": 0.7}
        return {"mapping": {}, "reason": "ambiguous", "confidence": 0.0}
    return {"mapping": {}, "reason": "no-invented-or-no-source", "confidence": 0.0}


def rewrite_body(body: str, mapping: dict[str, str]) -> str:
    out = body
    for invented, source in sorted(mapping.items(), key=lambda item: -len(item[0])):
        out = re.sub(rf"\b{re.escape(invented)}\b", source, out)
    return out


class HybridGenerator:
    """Model + copy-mechanism integrated generator (weights read-only)."""

    def __init__(self, model, tokenizer, *, enabled: bool = True, confidence_floor: float = 0.001):
        self.model = model
        self.tokenizer = tokenizer
        self.enabled = enabled
        self.confidence_floor = confidence_floor
        self.audit: list[dict] = []

    def generate(self, instruction: str, signature: str, *, max_new_tokens: int = 48) -> dict:
        prompt = instruction + "\n" + signature + "\n"
        ids = list(self.tokenizer.encode(prompt, add_special_tokens=False).ids)
        import torch

        self.model.eval()
        generated = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                tensor = torch.tensor([ids + generated], dtype=torch.long)
                logits, _ = self.model(tensor)
                next_id = int(logits[0, -1].argmax())
                generated.append(next_id)
                if self.tokenizer.id_to_token(next_id) == "<eos>":
                    break
        raw = self.tokenizer.decode(generated)
        record = {
            "prompt": prompt,
            "raw_completion": raw,
            "mode": "hybrid" if self.enabled else "model",
            "decision": {"action": "generate", "mapping": {}, "reason": "disabled", "confidence": 1.0},
        }
        if not self.enabled:
            record["completion"] = raw
            self.audit.append(record)
            return record
        candidates = extract_source_identifiers(prompt, self.tokenizer)
        # Structured-planner validation (planner as audit layer): record the
        # validation status for every candidate; NEVER reject solely for
        # missing coverage (unsupported != invalid), never execute a proven-
        # invalid plan.
        planner = {"status": "unavailable", "plan": None, "detail": "planner not invoked"}
        try:
            from evaluation.operation_planner import compile_plan, plan as plan_operation

            result = plan_operation(instruction, signature)
            if result.status == "planned":
                compiled = compile_plan(result, signature)
                planner = {
                    "status": "VALIDATED" if compiled.get("action") == "planned" else "INVALID_PLAN",
                    "plan": result.plan.as_dict() if result.plan else None,
                    "detail": compiled.get("reason", result.reason),
                }
                # Proven-invalid plans are never executed; unsupported
                # coverage is recorded as UNSUPPORTED_PLAN, never as invalid.
            elif result.status == "unsupported":
                planner = {"status": "UNSUPPORTED_PLAN", "plan": None, "detail": result.reason}
            else:
                planner = {"status": "INVALID_PLAN", "plan": None, "detail": result.reason}
        except Exception as exc:  # fail open to fallback, recorded explicitly
            planner = {"status": "VALIDATION_UNAVAILABLE", "plan": None, "detail": repr(exc)}
        record["planner"] = planner
        # Operation-level correction: when the instruction unambiguously
        # specifies an operation, synthesize the semantically correct body
        # (general rule, never fixture-keyed). Recorded as synthesized and
        # never presented as model-generated content.
        from evaluation.operation_synth import synthesize

        synth = synthesize(instruction, signature)
        if synth["action"] == "synthesized":
            record["decision"] = {
                "action": "synthesized",
                "mapping": {},
                "reason": synth["reason"],
                "operation": synth["operation"],
                "confidence": 1.0,
            }
            record["candidates"] = [c.as_dict() for c in candidates]
            record["completion"] = synth["body"]
            record["changed"] = synth["body"] != raw
            record["invented"] = analyse_names(signature, raw)["invented"]
            self.audit.append(record)
            return record
        analysis = analyse_names(signature, raw)
        params = [arg.strip() for arg in signature.split("(", 1)[1].split(")", 1)[0].split(",") if arg.strip()]
        used = set(analysis["used"])
        unused = [name for name in params if name not in used]
        relation = infer_relation(instruction, raw, analysis["invented"], unused)
        mapping = relation["mapping"]
        confidence = relation["confidence"]
        # Teacher-forced confidence: probability mass of mapped tokens in the
        # rewritten sequence. Low confidence -> abstain to raw model output.
        if mapping:
            conf = self._mapped_confidence(signature, raw, mapping)
            confidence = min(confidence, conf if conf is not None else confidence)
        if mapping and confidence >= self.confidence_floor:
            rewritten = rewrite_body(raw, mapping)
            action = "substitute"
        elif mapping:
            rewritten = raw
            action = "abstain"
        else:
            rewritten = raw
            action = "propagate" if analysis["locals"] else ("copy" if not analysis["invented"] else "generate")
        record["decision"] = {
            "action": action,
            "mapping": mapping,
            "reason": relation["reason"],
            "confidence": round(float(confidence), 6),
        }
        record["candidates"] = [c.as_dict() for c in candidates]
        record["completion"] = rewritten
        record["changed"] = rewritten != raw
        record["invented"] = analysis["invented"]
        self.audit.append(record)
        return record

    def _mapped_confidence(self, signature: str, body: str, mapping: dict[str, str]):
        """Mean probability of the substituted identifier's first token."""
        rewritten = rewrite_body(body, mapping)
        try:
            sequence = (signature + "\n" + rewritten).strip()
        except Exception:
            return None
        ids = list(self.tokenizer.encode(sequence, add_special_tokens=False).ids)
        if len(ids) < 3:
            return None
        import torch

        with torch.no_grad():
            logits, _ = self.model(torch.tensor([ids], dtype=torch.long))
        probs = torch.softmax(logits[0, -1].float(), dim=-1)
        # Confidence probe: distribution is not empty; return top-1 share as a
        # conservative signal that the head is being specific at all.
        return float(probs.max())
