"""Tests for TraceService (G5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.ingestion.storage.image_storage import ImageStorage
from src.observability.dashboard.services.trace_service import TraceService


@pytest.fixture()
def traces_file(tmp_path: Path) -> Path:
    p = tmp_path / "traces.jsonl"
    traces = [
        {
            "trace_id": "t1",
            "trace_type": "ingestion",
            "started_at": "2025-01-01T00:00:00",
            "elapsed_ms": 100.0,
            "stages": [
                {"stage": "load", "elapsed_ms": 20.0, "method": "pdf"},
                {"stage": "split", "elapsed_ms": 30.0},
                {"stage": "embed", "elapsed_ms": 50.0},
            ],
            "metadata": {"source": "a.pdf"},
        },
        {
            "trace_id": "t2",
            "trace_type": "query",
            "started_at": "2025-01-02T00:00:00",
            "elapsed_ms": 50.0,
            "stages": [
                {"stage": "dense_retrieval", "elapsed_ms": 25.0},
                {"stage": "rerank", "elapsed_ms": 25.0},
            ],
            "metadata": {},
        },
        {
            "trace_id": "t3",
            "trace_type": "ingestion",
            "started_at": "2025-01-03T00:00:00",
            "elapsed_ms": 200.0,
            "stages": [],
            "metadata": {},
        },
    ]
    with p.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    return p


class TestTraceService:

    def test_list_all(self, traces_file):
        svc = TraceService(traces_file)
        result = svc.list_traces()
        assert len(result) == 3
        # Newest first
        assert result[0]["trace_id"] == "t3"

    def test_list_by_type(self, traces_file):
        svc = TraceService(traces_file)
        ing = svc.list_traces(trace_type="ingestion")
        assert len(ing) == 2
        assert all(t["trace_type"] == "ingestion" for t in ing)

    def test_list_by_type_query(self, traces_file):
        svc = TraceService(traces_file)
        q = svc.list_traces(trace_type="query")
        assert len(q) == 1
        assert q[0]["trace_id"] == "t2"

    def test_list_with_limit(self, traces_file):
        svc = TraceService(traces_file)
        result = svc.list_traces(limit=1)
        assert len(result) == 1

    def test_get_trace_found(self, traces_file):
        svc = TraceService(traces_file)
        t = svc.get_trace("t1")
        assert t is not None
        assert t["trace_id"] == "t1"

    def test_get_trace_not_found(self, traces_file):
        svc = TraceService(traces_file)
        assert svc.get_trace("no_such") is None

    def test_get_stage_timings(self, traces_file):
        svc = TraceService(traces_file)
        t = svc.get_trace("t1")
        timings = svc.get_stage_timings(t)
        assert len(timings) == 3
        assert timings[0]["stage_name"] == "load"
        assert timings[0]["elapsed_ms"] == 20.0
        assert timings[0]["data"]["method"] == "pdf"

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.touch()
        svc = TraceService(p)
        assert svc.list_traces() == []

    def test_missing_file(self, tmp_path):
        svc = TraceService(tmp_path / "nonexistent.jsonl")
        assert svc.list_traces() == []

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"trace_id": "ok", "trace_type": "query", "started_at": ""}\nNOT_JSON\n')
        svc = TraceService(p)
        result = svc.list_traces()
        assert len(result) == 1
        assert result[0]["trace_id"] == "ok"

    def test_format_china_time_converts_utc_timestamp(self):
        result = TraceService.format_china_time("2026-06-26T13:45:11+00:00")
        assert result == "2026-06-26 21:45:11"

    def test_get_trace_images_returns_registered_image(self, tmp_path: Path):
        images_root = tmp_path / "images"
        image_path = images_root / "default" / "doc_image.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (24, 24), color="white").save(image_path)
        index_path = tmp_path / "image_index.db"
        ImageStorage(db_path=str(index_path), images_root=str(images_root)).register_image(
            "doc_image",
            image_path,
            collection="default",
            doc_hash="doc_hash",
            page_num=3,
        )
        trace = {
            "stages": [
                {
                    "stage": "upsert",
                    "data": {
                        "image_store": {
                            "images": [
                                {
                                    "image_id": "doc_image",
                                    "file_path": "/untrusted/path.png",
                                    "page": 3,
                                    "doc_hash": "doc_hash",
                                }
                            ]
                        }
                    },
                }
            ]
        }

        images = TraceService(
            tmp_path / "traces.jsonl",
            image_index_path=index_path,
            images_root=images_root,
        ).get_trace_images(trace)

        assert images == [
            {
                "image_id": "doc_image",
                "page_num": 3,
                "file_path": str(image_path.resolve()),
                "error": None,
            }
        ]

    def test_get_trace_images_reports_unregistered_trace_image(self, tmp_path: Path):
        trace = {
            "stages": [
                {
                    "stage": "upsert",
                    "data": {"image_store": {"images": [{"image_id": "missing"}]}},
                }
            ]
        }

        images = TraceService(
            tmp_path / "traces.jsonl",
            image_index_path=tmp_path / "missing.db",
            images_root=tmp_path / "images",
        ).get_trace_images(trace)

        assert images == [
            {
                "image_id": "missing",
                "page_num": None,
                "file_path": "",
                "error": "not registered in image index",
            }
        ]

    def test_get_trace_images_rejects_registered_path_outside_image_root(self, tmp_path: Path):
        images_root = tmp_path / "images"
        external_image = tmp_path / "external.png"
        Image.new("RGB", (24, 24), color="white").save(external_image)
        index_path = tmp_path / "image_index.db"
        ImageStorage(db_path=str(index_path), images_root=str(images_root)).register_image(
            "external",
            external_image,
            doc_hash="doc_hash",
        )
        trace = {
            "stages": [
                {
                    "stage": "upsert",
                    "data": {"image_store": {"images": [{"image_id": "external"}]}},
                }
            ]
        }

        images = TraceService(
            tmp_path / "traces.jsonl",
            image_index_path=index_path,
            images_root=images_root,
        ).get_trace_images(trace)

        assert images[0]["error"] == "image path is outside the configured image storage"

    def test_get_trace_images_uses_document_hash_for_old_trace(self, tmp_path: Path):
        images_root = tmp_path / "images"
        image_path = images_root / "default" / "legacy.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (24, 24), color="white").save(image_path)
        index_path = tmp_path / "image_index.db"
        ImageStorage(db_path=str(index_path), images_root=str(images_root)).register_image(
            "legacy",
            image_path,
            collection="default",
            doc_hash="legacy_hash",
            page_num=1,
        )
        trace = {
            "stages": [
                {"stage": "integrity", "data": {"file_hash": "legacy_hash"}},
                {"stage": "image_index", "data": {"count": 1}},
            ]
        }

        images = TraceService(
            tmp_path / "traces.jsonl",
            image_index_path=index_path,
            images_root=images_root,
        ).get_trace_images(trace)

        assert images[0]["image_id"] == "legacy"
        assert images[0]["page_num"] == 1
        assert images[0]["error"] is None
