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

from training.tokenizer_contract import (
    CONTRACT_V1,
    CONTRACT_V2,
    detect_contract,
    require_contract,
)

BOS = "<bos>"
EOS = "<eos>"

#: Contract used for all new training and serving. v1 remains renderable so the
#: historical M7 checkpoint can be reproduced and served through its documented
#: legacy path, but new artifacts must record v2.
CONTRACT_ID = CONTRACT_V2
LEGACY_CONTRACT_ID = CONTRACT_V1

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


def _render_v2(system: str, user: str, assistant: str | None) -> str:
    """Contract v2: explicit single-token role markers and an explicit end-of-turn.

    The prompt half is still a literal prefix of the completion; the assistant
    marker is emitted by the prompt so the model is conditioned to answer, and
    the completion adds the response plus the end-of-turn token it must learn.
    """
    text = (
        "<|system|>" + system + "<|end_of_turn|>"
        + "<|user|>" + user + "<|end_of_turn|>"
        + "<|assistant|>"
    )
    if assistant is not None:
        text += assistant + "<|end_of_turn|>"
    return text


def render_prompt(
    system: str, user: str, *, contract_id: str = LEGACY_CONTRACT_ID
) -> str:
    """Prompt half of the template: everything the model must not be trained to emit.

    ``contract_id`` defaults to the legacy v1 layout so historical callers keep
    their exact bytes. New code should call :func:`render_for`, which derives the
    contract from the tokenizer instead of assuming one.
    """
    if contract_id == CONTRACT_V2:
        return _render_v2(system, user, None)
    if contract_id == CONTRACT_V1:
        return _render(system, user, None)
    raise ValueError(f"unknown prompt contract: {contract_id!r}")


def render_completion(
    system: str, user: str, assistant: str, *, contract_id: str = LEGACY_CONTRACT_ID
) -> str:
    """Full training example under the requested prompt contract."""
    if contract_id == CONTRACT_V2:
        return _render_v2(system, user, assistant)
    if contract_id == CONTRACT_V1:
        return _render(system, user, assistant)
    raise ValueError(f"unknown prompt contract: {contract_id!r}")


def render_for(
    tokenizer, system: str, user: str, assistant: str | None = None
) -> str:
    """Render using the strongest contract ``tokenizer`` actually supports.

    This is the single entry point training and serving should use: the contract
    is a property of the tokenizer, not a caller opinion, so the two sides cannot
    disagree about which layout is in force.
    """
    contract = detect_contract(tokenizer)
    require_contract(tokenizer, contract)
    if assistant is None:
        return render_prompt(system, user, contract_id=contract)
    return render_completion(system, user, assistant, contract_id=contract)


def prompt_token_length(
    tokenizer, system: str, user: str, *, contract_id: str = LEGACY_CONTRACT_ID
) -> int:
    """Length of the prompt prefix in tokens, using the same renderer as training.

    Returned value is the count of *supervised* tokens to mask, consistent with
    ``input_ids = ids[:-1]`` / ``labels = ids[1:]``.
    """
    prompt = render_prompt(system, user, contract_id=contract_id)
    return len(tokenizer.encode(prompt).ids) - 1


def render_chat(tokenizer, messages, default_system: str | None = None) -> str:
    """Render an ordered role/content list under the tokenizer's contract.

    Serving must not invent a second layout for multi-turn input. v2 renders each
    role as its reserved single token; v1 cannot express roles or turns, so it is
    restricted to a single user turn and refuses rather than emitting delimiters
    the tokenizer would shred.
    """
    contract = detect_contract(tokenizer)
    require_contract(tokenizer, contract)

    def role_of(message):
        return message.get("role") if isinstance(message, dict) else getattr(message, "role", None)

    def content_of(message):
        return message.get("content") if isinstance(message, dict) else getattr(message, "content", "")

    system_parts = [content_of(m) for m in messages if role_of(m) == "system"]
    turns = [m for m in messages if role_of(m) in ("user", "assistant")]
    if not turns:
        raise ValueError("messages must contain at least one user or assistant turn")
    if role_of(turns[0]) != "user":
        raise ValueError("the first non-system turn must be a user turn")

    system = "\n".join(system_parts) if system_parts else (default_system or "")
    if contract == CONTRACT_V1:
        if len(turns) != 1:
            raise ValueError(
                "the legacy contract qm.plain_instruction/1 supports a single "
                "user turn only; multi-turn requires qm.chat_instruction/2"
            )
        return render_prompt(system, content_of(turns[0]), contract_id=CONTRACT_V1)

    parts = []
    if system:
        parts.append("<|system|>" + system + "<|end_of_turn|>")
    for turn in turns:
        marker = "<|user|>" if role_of(turn) == "user" else "<|assistant|>"
        parts.append(marker + content_of(turn) + "<|end_of_turn|>")
    parts.append("<|assistant|>")
    return "".join(parts)