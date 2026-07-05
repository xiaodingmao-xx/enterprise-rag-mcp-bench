"""Tests for Data Browser image preview validation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.observability.dashboard.pages.data_browser import _image_preview_error


def test_image_preview_error_returns_none_for_normal_image(tmp_path: Path) -> None:
    image_path = tmp_path / "normal.png"
    Image.new("RGB", (80, 40), color="white").save(image_path)

    assert _image_preview_error(image_path, preview_width=200) is None


def test_image_preview_error_detects_missing_file(tmp_path: Path) -> None:
    error = _image_preview_error(tmp_path / "missing.png")

    assert error == "file missing"


def test_image_preview_error_detects_too_wide_preview(tmp_path: Path) -> None:
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (1000, 1), color="white").save(image_path)

    error = _image_preview_error(image_path, preview_width=200)

    assert error is not None
    assert "aspect ratio too wide" in error


def test_image_preview_error_detects_invalid_image(tmp_path: Path) -> None:
    image_path = tmp_path / "broken.png"
    image_path.write_text("not an image", encoding="utf-8")

    error = _image_preview_error(image_path, preview_width=200)

    assert error is not None
    assert "invalid image" in error
