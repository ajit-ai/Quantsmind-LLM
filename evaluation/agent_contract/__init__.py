"""Offline contract tests for a coding-agent client against the Taksh server.

The harness in :mod:`evaluation.agent_contract.client` simulates the request
and response behaviour an external coding agent relies on, and drives it
against the real FastAPI application through an in-process transport. No
network is used and no external repository is touched.

Passing these tests establishes the **wire contract**. It does not establish
interoperability with QuantsMind-Agent, which lives elsewhere: the honest
status is ``EXTERNAL_AGENT_INTEGRATION_PENDING``.
"""

from evaluation.agent_contract.client import (
    AgentClientError,
    SimulatedCodingAgentClient,
    TakshTransport,
)

__all__ = ["AgentClientError", "SimulatedCodingAgentClient", "TakshTransport"]
