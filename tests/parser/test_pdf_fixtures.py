from __future__ import annotations

import pytest


def test_fixture_generator_is_available_without_importing_ocr_engines():
    module = pytest.importorskip("tests.fixtures.generate_pdf_fixtures")
    assert callable(module.generate_all)
