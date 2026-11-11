"""TAKSH DLCAE Runtime Foundation (L1.0.0-ENG).

Dynamic Language Comprehension and Adaptation Engine — operational runtime
foundation. This is an engineering qualification artifact only:

- No model weights exist for the canonical 111M configuration; the missing
  checkpoint remains missing.
- No training, fine-tuning or weight update is authorized (A1.1-C:
  preparation-only; A2-E: no compute).
- No capability claim is made anywhere in this package.
- Registration or discovery of a language never implies comprehension,
  adaptation or qualification.
- All persistence fixtures are synthetic; no user artifact is modified.

Package layout:

- ``types``        shared enums, canonical JSON, fingerprints
- ``comprehension`` Incremental Comprehension Engine (L1-L7 levels)
- ``sequences``    Sequence Orchestration Engine (DAG, checkpoint/resume)
- ``knowledge``    Knowledge Formation Engine (OBSERVED..INVALIDATED)
- ``registry``     dynamic language registry integration (no ceiling)
- ``adaptation``   Language Adaptation Engine (status promotion/demotion)
- ``validation``   Validation Engine (model-only vs tool-assisted, fail-closed)
- ``learning``     Learning Feedback Engine (authorization-gated handoff)
- ``persistence``  atomic, integrity-verified sequence/knowledge state
- ``karkain``      Karkain first-class adapter (KCC unavailable: BLOCKED)
- ``experiments``  A/B/C comparison framework (no experiments executed)
- ``runtime``      DLCAERuntime orchestration + eight-subsystem data flow
"""

__version__ = "0.1.0"

from taksh.dlcae.types import (  # noqa: F401
    CAPABILITY_STATES,
    NON_CAPABILITY_STATES,
    Observation,
    QualificationStatus,
    TaskState,
    ValidationVerdict,
    VerificationState,
    canonical_json,
    fingerprint,
)
