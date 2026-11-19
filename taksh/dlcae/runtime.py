"""DLCAE Runtime — orchestration and lifecycle contracts.

Wires the six engines into one deterministic runtime:

    source -> Incremental Comprehension -> Knowledge Formation
           -> Validation (model-only / tool-assisted) -> Learning Feedback
           -> (training handoff DISABLED under A1.1-C / A2-E)

plus dynamic language registry integration, sequence persistence/recovery,
and a deterministic eight-subsystem data-flow demonstration using test
doubles. The missing canonical 111M checkpoint stays missing; v180_sft is
never substituted; nothing here executes training.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from taksh.dlcae.adaptation import LanguageAdaptationEngine
from taksh.dlcae.comprehension import ComprehensionBundle, IncrementalComprehensionEngine
from taksh.dlcae.karkain import KarkainAdapter
from taksh.dlcae.knowledge import KnowledgeFormationEngine, provenance_from_source
from taksh.dlcae.learning import LearningFeedbackEngine
from taksh.dlcae.persistence import PersistenceStore
from taksh.dlcae.registry import DynamicLanguageRegistry
from taksh.dlcae.sequences import SequenceOrchestrationEngine
from taksh.dlcae.types import VerificationState, fingerprint, utcnow_iso
from taksh.dlcae.validation import ValidationEngine


class DLCAERuntime:
    """The DLCAE runtime foundation. Deterministic; filesystem state only."""

    def __init__(self, state_dir: Path | str, *,
                 persistence: PersistenceStore | None = None) -> None:
        self.state_dir = Path(state_dir)
        self.persistence = persistence or PersistenceStore(self.state_dir / "_persistence")
        self.validation = ValidationEngine()
        self.registry = DynamicLanguageRegistry()
        self.knowledge = KnowledgeFormationEngine()
        self.comprehension = IncrementalComprehensionEngine()
        self.sequences = SequenceOrchestrationEngine()
        self.adaptation = LanguageAdaptationEngine(self.registry, self.validation)
        self.learning = LearningFeedbackEngine()
        self.karkain = KarkainAdapter()
        self.sessions: dict[str, dict[str, Any]] = {}
        self._semantic_engine_instance = None   # L1.0.0-ENG2: lazily created

    # ------------------------------------------------------------------ #
    def open_session(self, document_id: str, language_id: str, text: str, *,
                     source: str = "session", license_spdx: str = "owner-internal-research") -> str:
        """Open a comprehension session; builds a tracked sequence."""
        session_id = f"session-{fingerprint({'doc': document_id, 'src': source, 'n': utcnow_iso()})[:12]}"
        sequence_id = self.sequences.create_sequence(
            f"comprehend:{document_id}",
            [{"task_id": "C", "action": "comprehend_document",
              "payload": {"document_id": document_id, "language_id": language_id,
                          "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}},
             {"task_id": "C1", "action": "form_knowledge", "depends_on": ["C"],
              "payload": {"document_id": document_id}}],
        )
        self.sessions[session_id] = {
            "session_id": session_id,
            "document_id": document_id,
            "language_id": language_id,
            "text": text,
            "source": source,
            "license_spdx": license_spdx,
            "sequence_id": sequence_id,
            "opened": utcnow_iso(),
            "bundle": None,
            "status": "OPEN",
        }
        handlers = self.sequences.handlers
        handlers["comprehend_document"] = self._handler_comprehend
        handlers["form_knowledge"] = self._handler_form_knowledge
        return session_id

    def _handler_comprehend(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._open_session_for(payload["document_id"])
        bundle = self.comprehension.comprehend(payload["document_id"], session["text"],
                                               payload["language_id"])
        session["bundle"] = bundle
        return {"level_status": dict(bundle.level_status),
                "verification_status": bundle.verification_status,
                "bundle_fingerprint": bundle.fingerprint()}

    def _handler_form_knowledge(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._open_session_for(payload["document_id"])
        bundle = session["bundle"]
        obs = bundle.observations()
        created = []
        for o in obs:
            state = VerificationState.INFERRED if o.state is VerificationState.INFERRED \
                else VerificationState.OBSERVED
            record = self.knowledge.add(
                session["language_id"], kind=_kind_for(o.kind),
                name=f"{o.kind}:L{o.level}",
                content={"detail": o.detail, "offsets": o.offsets,
                         "source_document": session["document_id"]},
                provenance=provenance_from_source(
                    session["source"], license_spdx=session["license_spdx"],
                    record_id=o.provenance.record_id,
                    source_kind=o.provenance.source_kind),
                state=state, confidence=o.confidence)
            created.append(record.record_id)
        session["status"] = "KNOWLEDGE_FORMED"
        return {"knowledge_records": created}

    # ------------------------------------------------------------------ #
    def run_session(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        result = self.sequences.run(session["sequence_id"])
        self._persist_session(session)
        session["status"] = result["state"]
        return result

    def validate_stage(self, stage: str, *, language_id: str, source: str,
                       mode: str = "model_only") -> dict[str, Any]:
        return self.validation.validate(stage, {"source": source},
                                        language_id=language_id, mode=mode)

    def make_learning_candidates(self, session_id: str) -> list[dict[str, Any]]:
        session = self._require_session(session_id)
        bundle = session.get("bundle")
        if bundle is None:
            return []
        candidates = []
        for observation in bundle.observations():
            verdict = self.validation.validate(
                "parser", {"source": session["text"]},
                language_id=session["language_id"], mode="model_only")
            candidate = self.learning.from_observation(
                observation.to_dict(), language_id=session["language_id"],
                source_provenance=provenance_from_source(
                    session["source"], license_spdx=session["license_spdx"],
                    record_id=observation.provenance.record_id,
                    source_kind=observation.provenance.source_kind),
                content_ref=f"{bundle.document_id}#{observation.observation_id}")
            self.learning.eligibility_check(candidate, validation_verdict=verdict["verdict"])
            candidates.append(self.learning.candidates[candidate["candidate_id"]])
        return candidates

    def handoff(self, candidate_ids: list[str] | None = None) -> dict[str, Any]:
        """Training handoff — always refused under A1.1-C / A2-E."""
        return self.learning.handoff_to_training(candidate_ids)

    # ------------------------------------------------------------------ #
    def subsystem_flow_demo(self, document_id: str, language_id: str, text: str) -> dict[str, Any]:
        """Deterministic eight-subsystem data flow with test doubles."""
        doubles = _EightSubsystemDoubles()
        source = doubles.ingest(document_id, language_id, text)   # Code Intelligence double
        comprehended = self.comprehension.comprehend(document_id, text, language_id)
        records = []
        for o in comprehended.observations():
            records.append(doubles.knowledge_formation(o.to_dict()))
        validated = [self.validation.validate("structural", {"source": text},
                                              language_id=language_id,
                                              mode="model_only")
                     for _ in records]
        evaluation = doubles.evaluation_record(validated)
        agent_action = doubles.agent_decide(evaluation)
        memory_write = doubles.memory_record({
            "document_id": document_id,
            "language_id": language_id,
            "levels": sorted(comprehended.levels),
            "observations": sum(len(v) for v in comprehended.levels.values()),
        })
        return {
            "schema": "taksh.dlcae.subsystem_flow/1",
            "at": utcnow_iso(),
            "code_intelligence": source,
            "dlcae": {"levels": {str(k): v for k, v in comprehended.level_status.items()},
                      "observations": sum(len(v) for v in comprehended.levels.values())},
            "knowledge_engine": {"records_formed": len(records)},
            "validation_engine": validated,
            "evaluation_engine": evaluation,
            "agent_engine": agent_action,
            "memory_engine": memory_write,
            "learning_engine": "handoff DISABLED (A1.1-C: preparation-only)",
            "engineering_engine": "no engineering action authorized; reporting only",
            "training": "DISABLED — no weights touched",
        }

    # ------------------------------------------------------------------ #
    def _open_session_for(self, document_id: str) -> dict[str, Any]:
        for session in sorted(self.sessions.values(), key=lambda s: s["opened"]):
            if session["document_id"] == document_id and session["status"] == "OPEN":
                return session
        raise KeyError(f"no open session for document: {document_id}")

    def _require_session(self, session_id: str) -> dict[str, Any]:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown session: {session_id}") from None

    def _persist_session(self, session: dict[str, Any]) -> None:
        bundle = session.get("bundle")
        if bundle is not None:
            self.persistence.save_json(
                f"sessions/{session['session_id']}.json",
                {"session": {k: v for k, v in session.items() if k != "text" and k != "bundle"},
                 "bundle": bundle.to_dict()})
        self.persistence.save_json(
            f"sequences/{session['sequence_id']}.json",
            self.sequences.checkpoint(session["sequence_id"]))

    # ------------------------------------------------------------------ #
    def semantic_analysis(self, document_id: str, language_id: str, text: str, *,
                          file: str | None = None, module: str | None = None,
                          mode: str = "model_only", stop_after: str | None = None,
                          resume_token: str | None = None,
                          extra_sources: dict[str, str] | None = None,
                          changed_modules: list[str] | None = None) -> dict[str, Any]:
        """Run the L1.0.0-ENG2 semantic pipeline (additive hook; no engines forked)."""
        return self._semantic_engine().analyze(
            document_id, language_id, text, file=file, module=module, mode=mode,
            stop_after=stop_after, resume_token=resume_token,
            extra_sources=extra_sources, changed_modules=changed_modules)

    def semantic_invalidate(self, source_sha256: str, reason: str) -> dict[str, Any]:
        """Invalidate every evidence binding to a source hash (ENG2 invalidation path)."""
        return self._semantic_engine().verify_source(source_sha256, reason=reason)

    def _semantic_engine(self):
        if self._semantic_engine_instance is None:
            from taksh.dlcae.semantic.engine import SemanticAnalysisEngine

            self._semantic_engine_instance = SemanticAnalysisEngine(
                knowledge=self.knowledge, validation=self.validation,
                state_store=self.persistence)
        return self._semantic_engine_instance

    def status(self) -> dict[str, Any]:
        integrity = self.persistence.verify_integrity()
        return {
            "schema": "taksh.dlcae.runtime_status/1",
            "at": utcnow_iso(),
            "sessions": sorted(self.sessions),
            "sequence_state": self.sequences.status_map(),
            "persistence_integrity": integrity,
            "knowledge_records": len(self.knowledge.all_records()),
            "learning_candidates": len(self.learning.candidates),
            "training_handoff": self.learning.handoff_to_training([])["authorized"],
            "canonical_checkpoint": "MISSING",
        }


def _kind_for(observation_kind: str) -> str:
    mapping = {
        "character_representation": "concept",
        "token_comprehension": "concept",
        "lexical_identifier": "language_rule",
        "syntax_structure": "syntax_structure",
        "parse_failure": "counterexample",
        "structural_heuristic": "syntax_structure",
        "semantic_type": "type_information",
        "task_intent": "concept",
        "execution_verification": "dependency_information",
    }
    return mapping.get(observation_kind, "concept")


class _EightSubsystemDoubles:
    """Deterministic doubles for the eight TAKSH subsystems (not models)."""

    def ingest(self, document_id, language_id, text):
        return {"accepted": True, "document_id": document_id,
                "language_id": language_id, "text_bytes": len(text.encode("utf-8"))}

    def knowledge_formation(self, observation):
        return {"formed": True, "observation_id": observation.get("observation_id")}

    def evaluation_record(self, verdicts):
        return {"records": len(verdicts),
                "passes": sum(1 for v in verdicts if v["verdict"] == "PASS"),
                "blocked": sum(1 for v in verdicts if v["verdict"] == "BLOCKED"),
                "capability_verdict": None}

    def agent_decide(self, evaluation):
        return {"action": "report_only",
                "reason": "no engineering action authorized; agent reports, never executes, under A1.1-C"}

    def memory_record(self, payload):
        return {"recorded": True, "record_sha256": fingerprint(payload)}
