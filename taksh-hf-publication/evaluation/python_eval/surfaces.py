"""Task surfaces for the v1.5.0 evaluation (Taksh).

v1.4.0 measured one mixed suite, so a base completion model was scored on
instruction-style prompts. That is a measurement mismatch, not a capability
result. v1.5.0 separates the surfaces explicitly and never converts one
surface into another.

``base_completion``
    The input is a code prefix. The model continues it. Correctness is
    judged on the **assembled program** (prefix + continuation), because a
    continuation that only makes sense with the prefix in front is still a
    correct continuation.

``instruction_to_code``
    The input is a natural-language instruction. The model answers with code;
    the code is extracted under rules declared in the case, and the extracted
    program is judged. Only a checkpoint that is qualified on instructions may
    be scored here.

``chat_instruction``
    The input is a natural-language instruction delivered through a chat
    prompt contract. The checkpoint must declare that contract as trained or
    qualified; otherwise the case is **refused**, not silently converted.
"""

from __future__ import annotations

SURFACES = ("base_completion", "instruction_to_code", "chat_instruction")

#: Which checkpoint capability each surface requires.
SURFACE_REQUIREMENTS = {
    "base_completion": "completion",
    "instruction_to_code": "instruction",
    "chat_instruction": "chat",
}


class SurfaceError(ValueError):
    """A surface and a checkpoint cannot be combined."""


def surface_of(case: dict) -> str:
    """The surface of a case, derived from the v1.4.0 `prompt_format`."""
    explicit = case.get("surface")
    if explicit:
        if explicit not in SURFACES:
            raise SurfaceError(f"unknown surface: {explicit!r}")
        return explicit
    prompt_format = case.get("prompt_format", "code_completion")
    if prompt_format == "code_completion":
        return "base_completion"
    return "instruction_to_code"


def check_contract(case: dict, capabilities: dict | None) -> tuple:
    """Return ``(allowed, reason)`` for using a case with a checkpoint.

    ``capabilities`` is the checkpoint's declared capability set, for example
    ``{"completion": True, "instruction": False, "chat": False}``. ``None``
    means the checkpoint declares nothing, which is treated as "not qualified"
    for the instruction surfaces and is fine for base completion.
    """
    surface = surface_of(case)
    required = SURFACE_REQUIREMENTS[surface]
    if surface == "base_completion":
        return True, ""
    capabilities = capabilities or {}
    if not capabilities.get(required, False):
        return False, (
            f"surface '{surface}' requires the checkpoint to be qualified on "
            f"'{required}'; the checkpoint declares {sorted(capabilities)}"
        )
    if surface == "chat_instruction":
        contract = case.get("prompt_contract")
        declared = capabilities.get("prompt_contract")
        if contract and declared and contract != declared:
            return False, (
                f"case requires prompt contract {contract!r} but the "
                f"checkpoint declares {declared!r}"
            )
    return True, ""


def assemble_program(case: dict, output: str) -> str:
    """The program to validate for a completion case.

    For base completion the prefix and the continuation are joined, because
    judging the continuation alone would reject correct continuations that
    depend on the prefix. For instruction surfaces the extracted code is
    returned unchanged.
    """
    surface = surface_of(case)
    if surface == "base_completion":
        prefix = case.get("prompt", "")
        return prefix + (output or "")
    return output or ""
