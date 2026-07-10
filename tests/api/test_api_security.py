"""External-service-free tests for the P1 API boundary."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.errors import APIError


def _settings(*, mode="local-dev", secret="test-secret", rate=120, timeout=30, body=10_000):
    return SimpleNamespace(
        api=SimpleNamespace(
            environment="test",
            auth=SimpleNamespace(
                mode=mode,
                allow_local_dev_without_auth=True,
                algorithm="HS256",
                secret=secret,
                issuer="rag-api",
                audience="rag-clients",
                leeway_seconds=0,
            ),
            rate_limit=SimpleNamespace(
                enabled=True,
                backend="memory",
                tenant=SimpleNamespace(requests=rate, window_seconds=60),
                user=SimpleNamespace(requests=rate, window_seconds=60),
            ),
            limits=SimpleNamespace(max_request_body_bytes=body),
            timeout=SimpleNamespace(
                query_seconds=timeout,
                retrieval_seconds=timeout,
                rerank_seconds=timeout,
                llm_seconds=timeout,
                ingestion_seconds=timeout,
            ),
            mcp_gateway=SimpleNamespace(enabled=True),
        ),
    )


def _token(tenant_id="tenant-a", user_id="alice", *, secret="test-secret", exp=None):
    header = {"alg": "HS256", "typ": "JWT"}
    claims = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "roles": ["member"],
        "iss": "rag-api",
        "aud": "rag-clients",
        "exp": exp if exp is not None else 4_000_000_000,
    }

    def encode(value):
        return base64.urlsafe_b64encode(
            json.dumps(value, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()

    first, second = encode(header), encode(claims)
    signature = hmac.new(secret.encode(), f"{first}.{second}".encode(), hashlib.sha256).digest()
    return f"{first}.{second}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


class _Query:
    async def query(self, payload, context):
        return {"answer": "ok", "tenant_id": context.tenant_id}


class _SlowQuery:
    async def query(self, payload, context):
        await asyncio.sleep(0.1)
        return {"answer": "too late"}


class _Documents:
    async def get(self, document_id, context):
        if document_id == "tenant-b-doc":
            raise APIError("TENANT_MISMATCH", "The resource belongs to another tenant")
        raise APIError("DOCUMENT_NOT_FOUND", "Document not found")

    async def summary(self, document_id, context):
        return await self.get(document_id, context)


class _Services:
    def __init__(self, query=None):
        self.query = query or _Query()
        self.documents = _Documents()
        self.collections = SimpleNamespace(list_collections=lambda context: [])
        self.ingestion = SimpleNamespace()


def test_jwt_auth_required_and_valid_request() -> None:
    app = create_app(_settings(mode="jwt"), services=_Services())
    client = TestClient(app)

    missing = client.post("/v1/query", json={"query": "hello"})
    assert missing.status_code == 401
    assert missing.json()["error_code"] == "AUTH_REQUIRED"

    valid = client.post(
        "/v1/query",
        json={"query": "hello"},
        headers={"Authorization": f"Bearer {_token()}"},
    )
    assert valid.status_code == 200
    assert valid.json()["tenant_id"] == "tenant-a"


def test_tenant_mismatch_is_shared_by_rest_and_mcp() -> None:
    app = create_app(_settings(mode="jwt"), services=_Services())
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {_token()}"}

    rest = client.get("/v1/documents/tenant-b-doc", headers=headers)
    mcp = client.post(
        "/v1/mcp/tools/call",
        json={"name": "get_document_summary", "arguments": {"doc_id": "tenant-b-doc"}},
        headers=headers,
    )
    assert rest.status_code == 403
    assert mcp.status_code == 403
    assert rest.json()["error_code"] == mcp.json()["error_code"] == "TENANT_MISMATCH"


def test_rate_limit_and_retry_after() -> None:
    app = create_app(_settings(rate=1), services=_Services())
    client = TestClient(app)
    assert client.post("/v1/query", json={"query": "one"}).status_code == 200
    response = client.post("/v1/query", json={"query": "two"})
    assert response.status_code == 429
    assert response.json()["error_code"] == "RATE_LIMITED"
    assert "retry-after" in response.headers


def test_query_timeout_and_body_limit_are_safe() -> None:
    app = create_app(_settings(timeout=0.01, body=20), services=_Services(_SlowQuery()))
    client = TestClient(app)
    timeout_response = client.post("/v1/query", json={"query": "hello"})
    assert timeout_response.status_code == 504
    assert timeout_response.json()["error_code"] == "RETRIEVAL_TIMEOUT"
    assert "Traceback" not in timeout_response.text
    assert "Authorization" not in timeout_response.text

    large = client.post("/v1/query", content=b"x" * 100, headers={"content-type": "application/json"})
    assert large.status_code == 413
    assert large.json()["error_code"] == "REQUEST_TOO_LARGE"


def test_health_readiness_is_distinct_from_liveness() -> None:
    app = create_app(_settings(), services=_Services(), readiness_check=lambda: False)
    client = TestClient(app)
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503


def test_production_local_dev_auth_is_rejected_at_startup() -> None:
    settings = _settings()
    settings.api.environment = "production"
    with pytest.raises(ValueError):
        create_app(settings, services=_Services())
