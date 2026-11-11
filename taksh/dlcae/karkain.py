"""Karkain first-class language adapter — contract and blocked-state handling.

Preserves the pinned integration contract:

- `.kark` source files;
- Karkain SPEC v1.1.0;
- KCC as the authoritative compiler/validation oracle;
- KARK-BENCH as its dedicated benchmark (**zero instantiated tasks**);
- shared TAKSH foundation model with Karkain-specific knowledge.

This host cannot run `kcc.exe` (Windows error 0xC0000135). Therefore:

- every tool-assisted verdict is BLOCKED;
- no functional or engineering qualification is claimed or implied;
- KCC is never bypassed (no emitted code is compiled or executed);
- no KARK-BENCH task exists and none is invented here;
- the Karkain repository is read-only context — never modified.
"""

from __future__ import annotations

from typing import Any

from taksh.dlcae.types import ValidationVerdict, utcnow_iso

KARKAIN_SPEC_PIN = "SPEC v1.1.0"
KARKAIN_COMPILER_PIN = "kcc v1.1.0 (windows/amd64, Stable Build; commit e612ede, branch feature/v1.2.0-native-independence)"
KCC_STARTUP_FAILURE = "0xC0000135"
KARK_BENCH_TASKS = 0
ADAPTATION_PHASE = "DISCOVERED"


class KarkainAdapter:
    """Adapter contract only; qualification is explicitly out of reach here."""

    language_id = "karkain"
    extensions = (".kark",)

    def __init__(self) -> None:
        self.blocked_events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def is_karkain_source(self, filename: str) -> bool:
        return str(filename).lower().endswith(self.extensions)

    def spec_pin(self) -> dict[str, str]:
        return {"karkain_spec_version": "1.1.0",
                "karkain_compiler": KARKAIN_COMPILER_PIN,
                "contact_warning": "Spec is pinned to v1.1.0; any update requires owner sign-off"}

    def toolchain_availability(self) -> dict[str, Any]:
        return {
            "karkain_cli": {"available": "PARTIAL",
                            "note": "help/config/target/explain/version exit 0 on this host"},
            "kcc": {"available": False,
                    "error": KCC_STARTUP_FAILURE,
                    "detail": "kcc.exe cannot start: a required DLL is missing; "
                              "compile-pipeline commands exit 3 (ExitCompile) with zero diagnostics"},
            "verdict": "BLOCKED",
        }

    def validate(self, source: str, *, stage: str = "parser") -> dict[str, Any]:
        """Karkain validation is always BLOCKED on this host — never PASS."""
        result = {
            "schema": "taksh.dlcae.karkain_validation/1",
            "language_id": self.language_id,
            "stage": stage,
            "verdict": ValidationVerdict.BLOCKED.value,
            "reason": f"KCC unavailable ({KCC_STARTUP_FAILURE}); "
                      f"no validation oracle on this host; bypass prohibited",
            "mode": "tool_assisted",
            "model_only": None,
            "source_bytes": len(source.encode("utf-8")),
            "at": utcnow_iso(),
        }
        self.blocked_events.append(result)
        return result

    def kark_bench_status(self) -> dict[str, Any]:
        return {
            "benchmark": "TAKSH-KARK-BENCH v1.0.0 (schema only)",
            "instantiated_tasks": KARK_BENCH_TASKS,
            "model_only_bar": "NOT DEFINED",
            "compiler_assisted_bar": "NOT DEFINED",
            "capability_claim": "none",
        }

    def qualification_status(self) -> dict[str, Any]:
        return {
            "language_id": self.language_id,
            "status": ADAPTATION_PHASE,
            "first_class": True,
            "corpus_records": 0,
            "evidence": [
                f"kcc.exe startup failure {KCC_STARTUP_FAILURE} — unresolved",
                "TAKSH-KARK-BENCH: 0 instantiated tasks",
                "Karkain corpus: 0 authorized training-eligible tokens",
            ],
            "note": "DISCOVERED is an inventory state, never a capability state",
        }

    def blocked_summary(self) -> dict[str, Any]:
        avail = self.toolchain_availability()
        return {
            "schema": "taksh.dlcae.karkain_contract/1",
            "language_id": self.language_id,
            "first_class": True,
            "extensions": list(self.extensions),
            "spec_pin": self.spec_pin(),
            "toolchain": avail,
            "blocked_events": len(self.blocked_events),
            "kark_bench": self.kark_bench_status(),
            "qualification": self.qualification_status(),
            "actions_prohibited": [
                "modify the Karkain repository, compiler, specification or benchmark",
                "bypass KCC",
                "invent Karkain corpus samples or benchmark results",
                "claim functional or engineering qualification",
            ],
        }
