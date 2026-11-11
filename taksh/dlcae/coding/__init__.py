"""TAKSH Coding Intelligence Integration (L1.0.0-ENG3).

End-to-end coding-task pipeline: task → model availability gate → generation
→ DLCAE comprehension → semantic validation → bounded repair loop → final
engineering verdict. Additive; no second inference engine, no duplicated
DLCAE runtime, no training.

Model capability states (exactly five):

    MODEL_UNAVAILABLE | MODEL_READY | GENERATION_COMPLETED
    GENERATION_FAILED | GENERATION_BLOCKED

A synthetic double or unavailable checkpoint is never represented as a
coding demonstration. A1.1-C / A2-E remain authoritative; the training
handoff stays disabled.
"""

from taksh.dlcae.coding.model import (  # noqa: F401
    ModelCapabilityState,
    ModelSession,
    StaticModelDouble,
    canonical_fingerprint,
    canonical_geometry_match,
)
from taksh.dlcae.coding.task import CodingTask, DevSuite, DevTaskTests, render_task_prompt  # noqa: F401
from taksh.dlcae.coding.evidence import EvidenceCategory, may_promote  # noqa: F401
from taksh.dlcae.coding.orchestrator import (  # noqa: F401
    CodingTaskOrchestrator,
    OrchestratorError,
    TaskState,
    extract_code,
)
