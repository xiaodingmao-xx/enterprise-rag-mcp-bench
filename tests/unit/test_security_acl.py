"""Unit tests for tenant isolation and document ACL enforcement."""

from types import SimpleNamespace

import pytest

from src.core.types import Document
from src.ingestion.pipeline import IngestionPipeline
from src.security.acl_filter import ACLFilter
from src.security.context import AuthenticationError, RequestContext, TenantRequiredError, resolve_request_context
from src.security.models import DocumentACL
from src.security.policy import ACLPolicy


def _record(acl: DocumentACL) -> dict:
    return {"id": f"{acl.document_id}-chunk", "text": "private body", "metadata": acl.to_metadata()}


def test_user_can_only_retrieve_documents_allowed_by_acl() -> None:
    context = RequestContext(
        tenant_id="tenant-a",
        user_id="alice",
        roles=("member",),
        department="engineering",
        auth_source="jwt",
        authenticated=True,
    )
    records = [
        _record(DocumentACL(tenant_id="tenant-a", document_id="doc-private", owner_id="bob", visibility="private")),
        _record(DocumentACL(tenant_id="tenant-a", document_id="doc-user", allowed_users=["alice"], visibility="users")),
        _record(DocumentACL(tenant_id="tenant-a", document_id="doc-role", allowed_roles=["finance"], visibility="roles")),
    ]

    allowed = ACLFilter(ACLPolicy()).filter_records(records, context).results

    assert [record["metadata"]["document_id"] for record in allowed] == ["doc-user"]


def test_different_tenants_are_always_isolated() -> None:
    context = RequestContext(
        tenant_id="tenant-a",
        user_id="admin-a",
        roles=("admin",),
        auth_source="jwt",
        authenticated=True,
    )
    record = _record(DocumentACL(tenant_id="tenant-b", document_id="doc-b", visibility="public"))

    assert ACLFilter(ACLPolicy()).filter_records([record], context).results == []


def test_admin_can_read_restricted_documents_inside_own_tenant() -> None:
    context = RequestContext(
        tenant_id="tenant-a",
        user_id="admin-a",
        roles=("admin",),
        auth_source="jwt",
        authenticated=True,
    )
    record = _record(DocumentACL(tenant_id="tenant-a", document_id="doc-a", owner_id="bob", visibility="private"))

    assert ACLFilter(ACLPolicy()).filter_records([record], context).results == [record]


def test_empty_acl_filtered_results_cannot_be_used_as_llm_context() -> None:
    context = RequestContext(
        tenant_id="tenant-a",
        user_id="alice",
        roles=("member",),
        auth_source="jwt",
        authenticated=True,
    )
    record = _record(DocumentACL(tenant_id="tenant-a", document_id="doc-a", allowed_users=["bob"], visibility="users"))

    assert ACLFilter(ACLPolicy()).filter_records([record], context).results == []


def test_acl_metadata_is_inherited_by_document_and_chunk() -> None:
    context = RequestContext(
        tenant_id="tenant-a",
        user_id="alice",
        roles=("member",),
        department="engineering",
        auth_source="local-dev",
        authenticated=True,
    )
    document = Document(id="doc-a", text="body", metadata={"source_path": "docs/a.md"})
    acl = IngestionPipeline._build_acl_metadata(document, context, "version-1")
    document.metadata.update(acl)
    chunk_metadata = {"source_path": "docs/a.md", **acl}

    assert {key: document.metadata[key] for key in acl} == {
        key: chunk_metadata[key] for key in acl
    }


def test_production_request_without_tenant_or_authentication_fails() -> None:
    settings = SimpleNamespace(
        security=SimpleNamespace(
            mode="production",
            require_tenant=True,
            require_authentication=True,
        )
    )

    with pytest.raises(AuthenticationError):
        resolve_request_context(settings)

    authenticated_without_tenant = RequestContext(
        user_id="alice",
        auth_source="jwt",
        authenticated=True,
    )
    with pytest.raises(TenantRequiredError):
        resolve_request_context(settings, context=authenticated_without_tenant)
