"""Permission and approval system for TAKSH engineering."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class OperationClass(str, Enum):
    READ = "read"
    CONTROLLED = "controlled"
    MUTATION = "mutation"
    RESTRICTED = "restricted"


@dataclass
class PermissionDecision:
    allowed: bool
    reason: str
    operation: str
    approved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ApprovalRecord:
    approval_id: str
    scope: str
    operation: str
    approved_at: float
    expires_at: float
    revoked: bool = False

    def active(self) -> bool:
        return (not self.revoked) and time.time() <= self.expires_at


class PermissionPolicy:
    def __init__(self) -> None:
        self.approvals: dict[str, ApprovalRecord] = {}
        self._default_allow = {OperationClass.READ.value, OperationClass.CONTROLLED.value}

    def evaluate(self, operation: str, approved: bool = False, arguments: dict[str, Any] | None = None) -> PermissionDecision:
        if operation in self._default_allow:
            return PermissionDecision(True, "allowed by default", operation, approved=True)
        if operation == OperationClass.MUTATION.value:
            if approved:
                return PermissionDecision(True, "mutation approved", operation, approved=True)
            return PermissionDecision(False, "mutation requires explicit approval", operation, approved=False)
        if operation == OperationClass.RESTRICTED.value:
            return PermissionDecision(False, "restricted operation is not allowed", operation, approved=False)
        return PermissionDecision(False, f"unknown operation class: {operation}", operation, approved=False)

    def grant(self, scope: str, operation: str, ttl_seconds: int = 300) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_id=f"{scope}:{operation}:{int(time.time())}",
            scope=scope,
            operation=operation,
            approved_at=time.time(),
            expires_at=time.time() + ttl_seconds,
        )
        self.approvals[record.approval_id] = record
        return record

    def revoke(self, approval_id: str) -> None:
        if approval_id in self.approvals:
            self.approvals[approval_id].revoked = True

    def audit(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.approvals.values()]
