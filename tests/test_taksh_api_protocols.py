"""Pure-contract tests for Taksh's FIM and tool text protocols.

No model, no torch, no network: these assert the bytes and the refusal
behaviour that the serving endpoints depend on.
"""

from __future__ import annotations

import json

import pytest

from serving.fim import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, render_fim
from serving.tool_protocol import (
    ToolProtocolError,
    parse_tool_call,
    render_tools,
    validate_tool_choice,
    validate_tools,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


# --------------------------------------------------------------------------- #
# FIM
# --------------------------------------------------------------------------- #
def test_fim_without_suffix_is_plain_completion():
    assert render_fim("def add(a, b):") == "def add(a, b):"


def test_fim_with_suffix_uses_the_documented_sentinels():
    rendered = render_fim("def add(a, b):\n    ", "\n")
    assert rendered.startswith(FIM_PREFIX)
    assert f"{FIM_SUFFIX}\n{FIM_MIDDLE}" in rendered


# --------------------------------------------------------------------------- #
# Tool declaration validation
# --------------------------------------------------------------------------- #
def test_valid_tool_is_normalised():
    tools = validate_tools(TOOLS)
    assert tools == [
        {
            "name": "read_file",
            "description": "Read a file",
            "parameters": TOOLS[0]["function"]["parameters"],
        }
    ]


def test_empty_tool_list_is_refused():
    with pytest.raises(ToolProtocolError):
        validate_tools([])


def test_non_function_tool_type_is_refused():
    with pytest.raises(ToolProtocolError):
        validate_tools([{"type": "retrieval", "function": {"name": "x"}}])


def test_missing_function_name_is_refused():
    with pytest.raises(ToolProtocolError):
        validate_tools([{"type": "function", "function": {}}])


def test_duplicate_tool_names_are_refused():
    with pytest.raises(ToolProtocolError):
        validate_tools([TOOLS[0], TOOLS[0]])


@pytest.mark.parametrize("choice", [None, "auto", "none"])
def test_supported_tool_choices(choice):
    assert validate_tool_choice(choice, [{"name": "read_file"}]) in ("auto", "none")


@pytest.mark.parametrize("choice", ["required", {"type": "function"}, 1])
def test_unhonourable_tool_choice_is_refused(choice):
    with pytest.raises(ToolProtocolError):
        validate_tool_choice(choice, [{"name": "read_file"}])


def test_rendered_tools_include_schema_and_format():
    rendered = render_tools(validate_tools(TOOLS))
    assert "read_file" in rendered
    assert "path" in rendered
    assert "<tool_call>" in rendered


# --------------------------------------------------------------------------- #
# Tool-call parsing
# --------------------------------------------------------------------------- #
def _parse(text):
    return parse_tool_call(text, validate_tools(TOOLS))


def test_plain_text_is_returned_unchanged():
    content, call, error = _parse("def add(a, b):\n    return a + b")
    assert content == "def add(a, b):\n    return a + b"
    assert call is None and error is None


def test_valid_block_is_parsed_into_openai_shape():
    content, call, error = _parse(
        'Sure.\n<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}'
        "</tool_call>"
    )
    assert content == "Sure."
    assert error is None
    assert call["type"] == "function"
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.py"}


def test_arguments_may_arrive_as_a_json_string():
    _, call, error = _parse(
        '<tool_call>{"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}'
        "</tool_call>"
    )
    assert error is None
    assert json.loads(call["function"]["arguments"]) == {"path": "a.py"}


def test_unknown_tool_is_not_invented():
    content, call, error = _parse(
        '<tool_call>{"name": "write_file", "arguments": {}}</tool_call>'
    )
    assert call is None
    assert "not declared" in error
    assert content.startswith("<tool_call>")


def test_invalid_json_is_reported_not_guessed():
    _, call, error = _parse("<tool_call>{not json}</tool_call>")
    assert call is None
    assert "not valid JSON" in error


def test_unclosed_block_is_reported():
    _, call, error = _parse('<tool_call>{"name": "read_file"}')
    assert call is None
    assert "not closed" in error


def test_first_block_is_the_one_parsed():
    _, call, error = _parse(
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
        '<tool_call>{"name": "read_file", "arguments": {"path": "b.py"}}</tool_call>'
    )
    assert error is None
    assert json.loads(call["function"]["arguments"]) == {"path": "a.py"}
