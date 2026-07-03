"""E2E coverage for multi-format ingest discovery via scripts/ingest.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scripts.ingest import discover_files


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGEST_SCRIPT = PROJECT_ROOT / "scripts" / "ingest.py"


def _run_ingest(path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(INGEST_SCRIPT),
            "--path",
            str(path),
            "--collection",
            "e2e_txt_md",
            "--dry-run",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


def test_ingest_txt_file_dry_run(tmp_path: Path) -> None:
    txt_file = tmp_path / "knowledge.txt"
    txt_file.write_text("enterprise knowledge base text", encoding="utf-8")

    result = _run_ingest(txt_file)

    assert result.returncode == 0
    assert "Supported extensions" in result.stdout
    assert "knowledge.txt" in result.stdout
    assert "Dry run" in result.stdout


def test_ingest_markdown_file_dry_run(tmp_path: Path) -> None:
    md_file = tmp_path / "README.md"
    md_file.write_text("# Title\n\nMarkdown body", encoding="utf-8")

    result = _run_ingest(md_file)

    assert result.returncode == 0
    assert "README.md" in result.stdout
    assert "Dry run" in result.stdout


def test_directory_discovery_collects_supported_and_skips_unsupported(tmp_path: Path) -> None:
    supported = [
        tmp_path / "a.pdf",
        tmp_path / "b.md",
        tmp_path / "c.txt",
        tmp_path / "d.html",
        tmp_path / "e.htm",
        tmp_path / "f.docx",
        tmp_path / "g.py",
        tmp_path / "h.js",
        tmp_path / "i.java",
    ]
    unsupported = tmp_path / "archive.bin"

    for file_path in supported + [unsupported]:
        file_path.write_text("sample", encoding="utf-8")

    discovered = discover_files(
        str(tmp_path),
        [".pdf", ".md", ".txt", ".html", ".htm", ".docx", ".py", ".js", ".java"],
    )
    discovered_names = {path.name for path in discovered}

    assert discovered_names == {path.name for path in supported}
    assert unsupported.name not in discovered_names
