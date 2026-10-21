"""Single source of truth for the chat/response template (C3).

The previous template was::

    "<|system|>{system}\\n<|user|>{instruction}\\n<|assistant|>{response}<|eos|>"

None of ``<|system|>``, ``<|user|>``, ``<|assistant|>`` or ``<|eos|>`` exist in
either shipped tokenizer. Encoding that string shredded the markers into ordinary
byte-level tokens (``['<|', 'system', '|>', 'h', 'i', ...]``), so the model was
never shown a real turn delimiter and could not learn to produce one.

This module fixes that without inventing vocabulary and without touching a
tokenizer:

* The template is expressed with ``<bos>`` and ``<eos>``, the boundary tokens the
  shipped tokenizers actually register (``qm_code_bpe_v0``: ids 1 and 2).
* :func:`validate_chat_template` refuses a tokenizer that does not register every
  marker, so this class of defect cannot be reintroduced silently.
* Training (``training/sft.py``) and inference (``serving/code_chat.py``) both
  render through these functions, so they cannot drift apart.

Known limitation, stated plainly: the vocabulary has **no role tokens**, so this
template does not mark who is speaking. It encodes "system preamble, then request,
then response, terminated by ``<eos>``". Adding real role markers would mean
extending the vocabulary, resizing the embedding matrix and invalidating existing
checkpoints; that is a deliberate, separate decision and is not taken here.
"""

from __future__ import annotations

BOS = "<bos>"
EOS = "<eos>"

#: Rendered as ``BOS + system + "\\n\\n" + instruction + "\\n\\n" + response + EOS``.
#: Placeholder names are kept as ``system``/``instruction``/``response`` so existing
#: callers that do ``TEMPLATE.format(system=..., instruction=..., response=...)``
#: keep working; only the surrounding markers changed.
CHAT_TEMPLATE = BOS + "{system}\n\n{instruction}\n\n{response}" + EOS


def _render(system: str, user: str, assistant: str | None) -> str:
    """Shared body. ``assistant is None`` omits both the response and the EOS.

    The prompt half must stay a literal prefix of the completion, otherwise
    label masking would supervise or hide the wrong tokens. Emitting the EOS in
    the prompt would both terminate early and break that prefix property.
    """
    text = BOS + system + "\n\n" + user + "\n\n"
    if assistant is not None:
        text += assistant + EOS
    return text

#: Markers that must exist as single tokens in the tokenizer.
REQUIRED_MARKERS = (BOS, EOS)

SYSTEM_PROMPT = (
    "You are Quantsmind Coder, an expert programming assistant specializing in "
    "Python and JavaScript/TypeScript. You help with code generation, debugging, "
    "code review, refactoring, and technical explanations.\n"
    "- Always produce correct, idiomatic code with proper error handling.\n"
    "- Explain your reasoning before providing code solutions.\n"
    "- When reviewing code, identify bugs, security issues, and performance problems.\n"
    "- Follow PEP 8 for Python and standard conventions for JavaScript/TypeScript.\n"
    "- If you are unsure about an API or library version, state your assumptions."
)


class IncompatibleTokenizer(RuntimeError):
    """The tokenizer cannot represent the chat template as single tokens."""


def validate_chat_template(tokenizer) -> dict:
    """Verify every template marker is a registered single token.

    Returns the resolved marker ids. Raises :class:`IncompatibleTokenizer` naming
    the missing markers, rather than letting them be shredded into byte tokens.
    """
    resolved = {}
    missing = []
    for marker in REQUIRED_MARKERS:
        token_id = tokenizer.token_to_id(marker)
        if token_id is None:
            missing.append(marker)
        else:
            resolved[marker] = int(token_id)
    if missing:
        raise IncompatibleTokenizer(
            "chat template markers are not registered as single tokens: "
            + ", ".join(missing)
            + ". They would be split into byte-level fragments, so the model "
            "could never learn a turn boundary. Either extend the tokenizer or "
            "choose markers it registers."
        )
    return resolved


def render_prompt(system: str, user: str) -> str:
    """Prompt half of the template: everything the model must not be trained to emit.

    Ends without ``<eos>`` so it is a literal prefix of :func:`render_completion`.
    """
    return _render(system, user, None)


def render_completion(system: str, user: str, assistant: str) -> str:
    """Full training example, ending with the ``<eos>`` the model must learn to emit."""
    return _render(system, user, assistant)


def prompt_token_length(tokenizer, system: str, user: str) -> int:
    """Length of the prompt prefix in tokens, using the same renderer as training.

    Returned value is the count of *supervised* tokens to mask, consistent with
    ``input_ids = ids[:-1]`` / ``labels = ids[1:]``.
    """
    return len(tokenizer.encode(render_prompt(system, user)).ids) - 1