"""Text protocol shim for OpenAI-style tool declarations.

The current checkpoints have no native tool-call tokens and no tool-use
training. This module therefore does the only honest thing a serving layer can
do:

1. **Render** the caller's tool schemas into the prompt as text, so a model
   that happens to emit the documented block is understood.
2. **Parse** that block back into the OpenAI ``tool_calls`` shape.
3. **Abstain** when the model does not emit a valid block. It never invents a
   call, never executes anything, and never reports ``finish_reason:
   "tool_calls"`` for unparseable text.

This is protocol plumbing, not a capability claim. The model is not expected to
call tools today; the tests assert the contract, not model behaviour.
"""

from __future__ import annotations

import json
import uuid

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
TOOL_PROTOCOL_ID = "taksh.tool_text/1"


class ToolProtocolError(ValueError):
    """The caller's ``tools`` declaration is malformed."""

    def __init__(self, message: str, *, param: str = "tools"):
        super().__init__(message)
        self.param = param


def validate_tools(tools) -> list[dict]:
    """Return normalised tool declarations or raise :class:`ToolProtocolError`."""
    if not isinstance(tools, list) or not tools:
        raise ToolProtocolError("'tools' must be a non-empty list.")
    normalised = []
    seen = set()
    for index, item in enumerate(tools):
        if not isinstance(item, dict):
            raise ToolProtocolError(f"tools[{index}] must be an object.")
        if item.get("type", "function") != "function":
            raise ToolProtocolError(
                f"tools[{index}].type must be 'function'; other tool kinds are "
                "not supported."
            )
        function = item.get("function")
        if not isinstance(function, dict):
            raise ToolProtocolError(f"tools[{index}].function must be an object.")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ToolProtocolError(
                f"tools[{index}].function.name must be a non-empty string."
            )
        if name in seen:
            raise ToolProtocolError(f"duplicate tool name {name!r}.", param="tools")
        seen.add(name)
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, dict):
            raise ToolProtocolError(
                f"tools[{index}].function.parameters must be an object."
            )
        description = function.get("description", "")
        if not isinstance(description, str):
            raise ToolProtocolError(
                f"tools[{index}].function.description must be a string."
            )
        normalised.append({
            "name": name,
            "description": description,
            "parameters": parameters,
        })
    return normalised


def validate_tool_choice(tool_choice, tools) -> str:
    """Return ``"auto"`` or ``"none"``; refuse choices the server cannot honour."""
    if tool_choice is None:
        return "auto"
    if tool_choice == "auto":
        return "auto"
    if tool_choice == "none":
        return "none"
    raise ToolProtocolError(
        "only tool_choice='auto' and tool_choice='none' are supported; a forced "
        "or required tool call cannot be guaranteed by this model, so accepting "
        "it would be a false promise.",
        param="tool_choice",
    )


def render_tools(tools: list[dict]) -> str:
    """A prompt block describing the tools and the exact reply format."""
    schemas = json.dumps(
        [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
            for tool in tools
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return (
        "\n\nAvailable tools (JSON):\n"
        + schemas
        + "\n\nYou may call at most one tool. To call a tool, reply with exactly "
        "one block in this form:\n"
        + TOOL_OPEN
        + '{"name": "<tool_name>", "arguments": {<json arguments>}}'
        + TOOL_CLOSE
        + "\nIf no tool is needed, answer normally and do not emit a tool block."
    )


def _coerce_arguments(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("arguments must be a JSON object")
    return value


def parse_tool_call(text: str, tools: list[dict]) -> tuple[str | None, dict | None, str | None]:
    """Parse the first valid tool block in ``text``.

    Returns ``(content, tool_call, error)``:

    * ``content`` is the surrounding prose, or ``None`` when empty.
    * ``tool_call`` is the OpenAI-shaped call, or ``None``.
    * ``error`` is a diagnostic when a block was present but unusable.
    """
    if TOOL_OPEN not in text:
        return text, None, None

    allowed = {tool["name"] for tool in tools}
    start = text.find(TOOL_OPEN)
    end = text.find(TOOL_CLOSE, start + len(TOOL_OPEN))
    if end == -1:
        return text, None, "tool block is not closed"

    raw = text[start + len(TOOL_OPEN):end].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return text, None, f"tool block is not valid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return text, None, "tool block must be a JSON object"

    name = payload.get("name")
    if not isinstance(name, str) or name not in allowed:
        return text, None, f"tool {name!r} was not declared"
    try:
        arguments = _coerce_arguments(payload.get("arguments"))
    except (json.JSONDecodeError, ValueError) as exc:
        return text, None, f"tool arguments are invalid: {exc}"

    surrounding = (text[:start] + text[end + len(TOOL_CLOSE):]).strip()
    return (
        surrounding or None,
        {
            "id": f"call_{uuid.uuid4().hex}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
            },
        },
        None,
    )
