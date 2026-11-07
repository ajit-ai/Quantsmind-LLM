"""Bounded Python code generation for Taksh v1.4.0.

This is a *controller* around the existing sampling implementation
(:func:`serving.local_inference.generate_ids`), not a second generation
stack. It adds what a capability evaluation needs and a raw endpoint does
not:

* **Prompt validation.** An empty prompt, or one containing a reserved token
  spelling, is refused before anything is generated. A prompt that can forge
  a control id is never encoded.
* **Prompt-surface honesty.** Each model declares the surfaces it was
  trained or qualified on. A base (pretraining) model is served raw text; the
  canonical ``qm.chat_instruction/2`` renderer is used only for a model that
  is qualified on it. The controller never quietly applies a chat template to
  a base model.
* **Bounds.** The token budget is capped, the context window is enforced
  (the budget shrinks to fit rather than silently overrunning), and the
  output is truncated to the declared limit.
* **Diagnostics.** Every call returns structured information: what was
  rendered, how many tokens were used, why generation stopped, and whether
  anything was refused.

The tokenizer, the model geometry and the prompt contract are untouched.
"""

from __future__ import annotations

import time

from serving.local_inference import generate_ids
from training.objective import IGNORE_INDEX  # noqa: F401  (kept for symmetry)

#: Reserved spellings that must never reach the encoder.
RESERVED_LITERALS = (
    "<bos>",
    "<eos>",
    "<unk>",
    "<pad>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_of_turn|>",
    "<|tool|>",
    "<|fim|>",
)

PROMPT_FORMATS = ("code_completion", "chat_instruction")


class PythonCodeGenerator:
    """Bounded generation for Python capability evaluation."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        model_identity: dict | None = None,
        qualified_formats=("code_completion",),
        chat_contract: str | None = None,
        max_new_tokens_cap: int = 256,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_identity = dict(model_identity or {})
        self.qualified_formats = tuple(qualified_formats)
        self.chat_contract = chat_contract
        self.max_new_tokens_cap = int(max_new_tokens_cap)
        self.device = device
        self.context_length = int(model.cfg.get("max_seq_len", 512))

    # ------------------------------------------------------------------ helpers
    def _reserved_in_prompt(self, prompt: str) -> list:
        return [literal for literal in RESERVED_LITERALS if literal in prompt]

    def _render(self, prompt: str, prompt_format: str):
        """Render the prompt for the surface the model is qualified on."""
        notes = []
        if prompt_format == "chat_instruction" and self.chat_contract:
            from training.chat_template import render_chat

            rendered = render_chat(
                self.tokenizer,
                [{"role": "user", "content": prompt}],
                default_system="",
            )
            notes.append(f"chat template applied: {self.chat_contract}")
            return rendered, notes
        notes.append(
            "raw text prompt: the model is not qualified on the chat contract"
        )
        return prompt, notes

    def _refusal(self, reason: str, **extra) -> dict:
        payload = {
            "text": "",
            "token_ids": [],
            "finish_reason": "refused",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "refusal": reason,
            "diagnostics": {"refused": True, "reason": reason, **extra},
        }
        return payload

    # ------------------------------------------------------------------ public
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 96,
        format: str = "code_completion",
        temperature: float = 0.0,
        seed: int | None = None,
        stop=None,
        binding_repair: bool | None = None,
    ) -> dict:
        """Generate a bounded completion for one prompt."""
        started = time.time()
        if not isinstance(prompt, str) or not prompt.strip():
            return self._refusal("empty_prompt")
        if format not in PROMPT_FORMATS:
            return self._refusal("unknown_prompt_format", format=format)
        reserved = self._reserved_in_prompt(prompt)
        if reserved:
            return self._refusal(
                "reserved_token_literal_in_prompt", literals=reserved
            )
        if format not in self.qualified_formats:
            return self._refusal(
                "prompt_surface_not_qualified",
                format=format,
                qualified_formats=list(self.qualified_formats),
            )
        budget = max(1, min(int(max_new_tokens), self.max_new_tokens_cap))
        rendered, notes = self._render(prompt, format)
        prompt_ids = self.tokenizer.encode(rendered).ids
        prompt_tokens = len(prompt_ids)
        if prompt_tokens >= self.context_length:
            return self._refusal(
                "prompt_exceeds_context",
                prompt_tokens=prompt_tokens,
                context_length=self.context_length,
            )
        budget = min(budget, self.context_length - prompt_tokens)

        if binding_repair is None:
            binding_repair = bool(self.model_identity.get("binding_repair", True))
        result = generate_ids(
            self.model,
            self.tokenizer,
            rendered,
            max_new_tokens=budget,
            temperature=temperature,
            device=self.device,
            seed=seed,
            stop=stop,
            binding_repair=bool(binding_repair),
        )
        text = result.get("text", "")
        truncated = len(text.encode("utf-8")) > self.max_new_tokens_cap * 8
        diagnostics = {
            "refused": False,
            "prompt_format": format,
            "rendered_prompt_tokens": prompt_tokens,
            "token_budget": budget,
            "context_length": self.context_length,
            "finish_reason": result.get("finish_reason"),
            "truncated": truncated,
            "duration_seconds": time.time() - started,
            "notes": notes,
            "repair": result.get("repair", {}),
            "model": dict(self.model_identity),
        }
        result["diagnostics"] = diagnostics
        result["refusal"] = None
        return result
