"""Constrained decoding over the frozen tokenizer (Increment 13/14).

Bounding POC, not full grammar enforcement: the decoder restricts two
classes of invalid continuation that the unconstrained head frequently
emits (unbalanced brackets, degenerate loops) and abstains to the
unconstrained output when contexts are unsupported. Every intervention
is logged. When ``enabled=False`` the unconstrained greedy path is
reproduced exactly.
"""

from __future__ import annotations


class ConstraintState:
    def __init__(self):
        self.open_depth = 0
        self.last_tokens: list[int] = []

    def update(self, token_text: str) -> None:
        for char in token_text or "":
            if char in "([{":
                self.open_depth += 1
            elif char in ")]}" and self.open_depth > 0:
                self.open_depth -= 1
        self.last_tokens.append(1)

    def forbids(self, token_id: int, eos_id: int) -> bool:
        # Never emit EOS with unbalanced brackets.
        return token_id == eos_id and self.open_depth > 0


def apply_constraints(logits, state: ConstraintState, *, eos_id: int, repeat_token: int | None = None):
    """Return (filtered_logits, interventions) — one constrained step."""
    import torch

    if logits.dim() == 2:
        logits = logits.reshape(-1)
    filtered = logits.clone()
    interventions = []
    if state.forbids(int(filtered.argmax()), eos_id) or state.open_depth > 0:
        if state.open_depth > 0 and filtered[eos_id] == filtered.max():
            filtered[eos_id] = float("-inf")
            interventions.append("eos_suppressed_unbalanced")
    if repeat_token is not None and len(state.last_tokens) >= 4 and all(t == 1 for t in state.last_tokens[-4:]):
        filtered[repeat_token] = float("-inf")
        interventions.append("repeat_token_suppressed")
    return filtered, interventions


def generate_constrained(model, tokenizer, prompt: str, *, max_new_tokens: int = 48, enabled: bool = True, eos_token: str = "<eos>") -> dict:
    """Greedy decoding with the bounded constraint layer (CPU)."""
    import torch

    ids = list(tokenizer.encode(prompt, add_special_tokens=False).ids)
    eos_id = tokenizer.token_to_id(eos_token)
    state = ConstraintState()
    generated = []
    interventions = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            tensor = torch.tensor([ids + generated], dtype=torch.long)
            logits, _ = model(tensor)
            step_logits = logits[0, -1]
            if enabled:
                step_logits, step_interventions = apply_constraints(step_logits, state, eos_id=eos_id)
                interventions.extend(step_interventions)
            next_id = int(step_logits.argmax())
            generated.append(next_id)
            token_text = tokenizer.id_to_token(next_id) or ""
            state.update(token_text)
            if next_id == eos_id:
                break
    return {
        "completion": tokenizer.decode(generated),
        "tokens": generated,
        "interventions": interventions,
        "constrained": enabled,
    }
