"""Bridge model providers into the TAKSH change-planning orchestration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from serving.providers import (
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    ProviderCapabilities,
    ProviderModelIdentity,
)
from taksh.engineering import ChangePlan, FileChange

_SAFE_TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt", ".cfg", ".ini"}
_SECRET_NAME_PARTS = (".env", "secret", "credential", "token", "id_rsa", "id_ed25519", ".pem", ".key")
_MAX_FILE_CHARS = 4000
_MAX_FILES_WITH_CONTENTS = 5


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
            configured=True,
            status="test_only",
            test_only=True,
            modes=("structured-patch",),
            max_context_tokens=8192,
            notes="Deterministic structured patch fixture for conformance tests.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
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

    def __init__(
        self,
        model_provider: Any,
        *,
        allow_file_contents: bool = False,
        max_context_tokens: int = 8192,
    ) -> None:
        self.model_provider = model_provider
        self.allow_file_contents = allow_file_contents
        self.max_context_tokens = max_context_tokens
        self.name = getattr(model_provider, "name", "model-provider")

    @staticmethod
    def _relevant_files(task: Any, context: dict[str, Any]) -> list[str]:
        symbols = context.get("symbols", {})
        indexed = symbols.get("symbols", {}) if isinstance(symbols, dict) else {}
        candidates = list(indexed) if isinstance(indexed, dict) else []
        if not candidates:
            tests = context.get("tests", [])
            candidates = list(tests) if isinstance(tests, list) else []
        words = {word.lower() for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", task.objective)}
        scored: list[tuple[int, str]] = []
        for path in candidates:
            path_text = str(path)
            file_symbols = indexed.get(path, []) if isinstance(indexed, dict) else []
            names = {
                str(item.get("name", "")).lower()
                for item in file_symbols
                if isinstance(item, dict)
            }
            score = sum(1 for word in words if word and (word in path_text.lower() or word in names))
            scored.append((score, path_text))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [path for _, path in scored[:10]]

    def _file_context(self, task: Any, files: list[str]) -> tuple[str, bool]:
        if not self.allow_file_contents or not files:
            return "", False
        root = Path(task.root)
        blocks: list[str] = []
        transmitted = False
        for relative in files[:_MAX_FILES_WITH_CONTENTS]:
            if any(part in relative.lower() for part in _SECRET_NAME_PARTS):
                continue
            path = root / relative
            if path.suffix.lower() not in _SAFE_TEXT_SUFFIXES or not path.is_file():
                continue
            try:
                if path.stat().st_size > _MAX_FILE_CHARS * 4:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")[:_MAX_FILE_CHARS]
            except OSError:
                continue
            blocks.append(f"--- {relative} ---\n{text}")
            transmitted = True
        return "\n\n".join(blocks), transmitted

    def _prompt(self, task: Any, context: dict[str, Any]) -> tuple[str, bool]:
        summary = context.get("summary", {})
        relevant = self._relevant_files(task, context)
        contents, transmitted = self._file_context(task, relevant)
        tests = context.get("tests", [])
        lines = [
            "You are TAKSH's coding-agent provider bridge.",
            "Return only structured patch blocks using FILE/OPERATION/REASON/CONTENT/END.",
            "Do not emit shell commands. Paths must be relative to the project root.",
            f"Objective: {task.objective}",
            f"Root: {task.root}",
            f"Files: {summary.get('total_files')}; Python files: {summary.get('python_files')}",
            f"Relevant files: {', '.join(relevant) if relevant else '(none indexed)'}",
            f"Tests: {', '.join(str(t) for t in tests[:20]) if tests else '(none discovered)'}",
        ]
        if contents:
            lines.extend(["", "Bounded file context (explicitly authorized):", contents])
        return "\n".join(lines), transmitted

    def generate_changes(self, task: Any, context: dict[str, Any]) -> ChangePlan:
        prompt, transmitted = self._prompt(task, context)
        max_chars = max(1000, self.max_context_tokens * 4)
        request = GenerationRequest(
            prompt[:max_chars],
            GenerationConfig(max_new_tokens=512, seed=1),
        )
        result = self.model_provider.generate(request)
        changes = parse_structured_changes(result.text)
        identity = (
            self.model_provider.identity().as_dict()
            if hasattr(self.model_provider, "identity")
            else {"provider": self.name}
        )
        return ChangePlan(
            task_id=task.task_id,
            root=task.root,
            changes=changes,
            summary="structured model-provider proposal",
            provider=self.name,
            provenance={
                "provider": identity.get("provider"),
                "model_id": identity.get("model_id"),
                "revision": identity.get("revision"),
                "deterministic": bool(result.metadata.get("deterministic"))
                if isinstance(result.metadata, dict)
                else False,
                "file_contents_transmitted": bool(transmitted),
            },
        )


def parse_structured_changes(text: str) -> list[FileChange]:
    changes: list[FileChange] = []
    pattern = re.compile(
        r"FILE:\s*(?P<path>[^\n]+)\n"
        r"OPERATION:\s*(?P<operation>[^\n]+)\n"
        r"REASON:\s*(?P<reason>[^\n]+)\n"
        r"CONTENT:\n(?P<content>.*?)\nEND",
        re.S,
    )
    for match in pattern.finditer(text):
        changes.append(
            FileChange(
                path=match.group("path").strip(),
                operation=match.group("operation").strip(),
                content=match.group("content"),
                reason=match.group("reason").strip(),
            )
        )
    return changes
