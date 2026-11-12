"""Focused tests for the engineering REST API security boundary."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from serving.engineering_api import build_router


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TAKSH_ENGINEERING_TOKEN", "test-token")
    monkeypatch.setenv("TAKSH_TOKEN_SCOPES", "read,execute,mutate,memory,audit")
    app = FastAPI()
    app.include_router(build_router())
    return TestClient(app)


def test_health_is_public(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_protected_endpoint_denies_when_token_not_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("TAKSH_ENGINEERING_TOKEN", raising=False)
    monkeypatch.delenv("QM_API_KEY", raising=False)
    app = FastAPI()
    app.include_router(build_router())
    c = TestClient(app)
    response = c.get("/v1/projects/inspect", params={"root": str(tmp_path)})
    assert response.status_code == 503


def test_protected_endpoint_rejects_missing_token(client, tmp_path):
    response = client.get("/v1/projects/inspect", params={"root": str(tmp_path)})
    assert response.status_code == 401


def test_protected_endpoint_allows_configured_token(client, tmp_path):
    response = client.get("/v1/projects/inspect",
                          params={"root": str(tmp_path)},
                          headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert "summary" in response.json()


def test_insufficient_scope_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("TAKSH_ENGINEERING_TOKEN", "test-token")
    monkeypatch.setenv("TAKSH_TOKEN_SCOPES", "read")
    app = FastAPI()
    app.include_router(build_router())
    c = TestClient(app)
    response = c.get("/v1/audit", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 403


def test_apply_requires_mutate_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("TAKSH_ENGINEERING_TOKEN", "test-token")
    monkeypatch.setenv("TAKSH_TOKEN_SCOPES", "read,execute")
    app = FastAPI()
    app.include_router(build_router())
    c = TestClient(app)
    response = c.post("/v1/tasks/nope/apply", json={"plan": {"changes": []}, "approved": True},
                      headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 403


def test_audit_records_authorization_decisions(client, tmp_path):
    client.get("/v1/projects/inspect", params={"root": str(tmp_path)},
               headers={"Authorization": "Bearer test-token"})
    response = client.get("/v1/audit", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    records = response.json()["records"]
    assert any(record["decision"] == "allow" for record in records)
