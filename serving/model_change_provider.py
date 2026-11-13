"""Bridge model providers into the TAKSH change-planning orchestration."""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Iterable

from serving.providers import GenerationConfig, GenerationRequest, ProviderModelIdentity, ProviderCapabilities
from taksh.engineering import ChangePlan, FileChange


class DeterministicPatchProvider:
    """Test provider that emits structured patch text.

    This is not a language model and is not evidence of coding capability.
    """

    name = "deterministic-patch"

    def __init__(self, changes: list[FileChange]) -> None:
        self.changes = list(changes)
        self._identity = ProviderModelIdentity(
            provider=self.name,
            model_id="deterministic-patch-1",
            revision="test-double",
            source="structured-patch-fixture",
            license_or_rights="test double; no model rights claimed",
        ).validated()

    def identity(self) -> ProviderModelIdentity:
        return self._identity

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.name,
            available=True,
            offline=True,
            modes=("structured-patch",),
            max_context_tokens=8192,
            notes="Deterministic structured patch fixture for conformance tests.",
        )

    def generate(self, request: GenerationRequest):
        request.validated()
        text = ""
        for change in self.changes:
            text += (
                f"FILE: {change.path}\n"
                f"OPERATION: {change.operation}\n"
                f"REASON: {change.reason}\n"
                "CONTENT:\n"
                f"{change.content or ''}\n"
                "END\n"
            )
        from serving.providers import GenerationResult

        return GenerationResult(
            provider=self.name,
            model_id=self._identity.model_id,
            revision=self._identity.revision,
            text=text,
            raw_text=text,
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(text.split()),
            latency_seconds=0.0,
            repair={"status": "test-double"},
            metadata={"deterministic": True},
        )


class ModelChangeProvider:
    """Wraps a model provider and parses structured change proposals."""

    def __init__(self, model_provider, *, allow_file_contents: bool = False, max_context_tokens: int = 8192) -> None:
        self.model_provider = model_provider
        self.allow_file_contents = allow_file_contents
        self.max_context_tokens = max_context_tokens
        self.name = getattr(model_provider, "name", "model-provider")

    def _prompt(self, task, context: dict[str, Any]) -> str:
        summary = context.get("summary", {})
        lines = [
            "You are TAKSH's coding-agent provider bridge.",
            "Return only structured patch blocks using FILE/OPERATION/REASON/CONTENT/END.",
            f"Objective: {task.objective}",
            f"Root: {task.root}",
            f"Files: {summary.get('total_files')}; Python files: {summary.get('python_files')}",
        ]
        return "\n".join(lines)

    def generate_changes(self, task, context: dict[str, Any]) -> ChangePlan:
        request = GenerationRequest(
            self._prompt(task, context),
            GenerationConfig(max_new_tokens=512, seed=1),
        )
        result = self.model_provider.generate(request)
        changes = parse_structured_changes(result.text)
        identity = self.model_provider.identity().as_dict() if hasattr(self.model_provider, "identity") else {"provider": self.name}
        return ChangePlan(
            task_id=task.task_id,
            root=task.root,
            changes=changes,
            summary="structured model-provider proposal",
            provider=self.name,
            provenance={
                "provider": identity.get("provider"),
                "model_id": identity.get("model_id"),
                "deterministic": bool(result.metadata.get("deterministic")) if isinstance(result.metadata, dict) else False,
                "file_contents_transmitted": bool(self.allow_file_contents),
            },
        )


def parse_structured_changes(text: str) -> list[FileChange]:
    changes: list[FileChange] = []
    pattern = re.compile(r"FILE:\s*(?P<path>[^\n]+)\nOPERATION:\s*(?P<operation>[^\n]+)\nREASON:\s*(?P<reason>[^\n]+)\nCONTENT:\n(?P<content>.*?)\nEND", re.S)
    for match in pattern.finditer(text):
        changes.append(FileChange(
            path=match.group("path").strip(),
            operation=match.group("operation").strip(),
            content=match.group("content"),
            reason=match.group("reason").strip(),
        ))
    return changes
