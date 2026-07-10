"""Dependency-light table extraction with optional pdfplumber enhancement."""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from src.libs.loader.parsed_document import ExtractionQuality, Table, TableCell
from src.libs.parser.base import BaseParser


class TableParser(BaseParser):
    WARNING_DEPENDENCY_MISSING = "TABLE_PARSER_DEPENDENCY_MISSING"
    WARNING_FAILED = "TABLE_EXTRACTION_FAILED"

    def __init__(self, *, engine: str = "auto", extract_markdown: bool = True) -> None:
        self.engine = str(engine).lower()
        self.extract_markdown = bool(extract_markdown)
        self.last_warnings: List[str] = []

    def parse(self, file_path: str | Path, **kwargs: Any):
        from src.libs.parser.base import BasicTextParser

        path = Path(file_path).resolve()
        document_id = str(kwargs.get("document_id") or self._document_id(path))
        try:
            import fitz
        except ImportError:
            return BasicTextParser().parse(path, document_id=document_id)

        all_tables: List[Table] = []
        all_warnings: List[str] = []
        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                all_tables.extend(
                    self.parse_page(page, page_number=page_number, document_id=document_id)
                )
                all_warnings.extend(self.last_warnings)
        self.last_warnings = list(dict.fromkeys(all_warnings))
        return self._to_parsed_document(path, document_id, all_tables)

    def parse_page(self, page: Any, *, page_number: int, document_id: str) -> List[Table]:
        self.last_warnings = []
        if self.engine in {"pdfplumber", "camelot"} and not importlib.util.find_spec(self.engine):
            self.last_warnings.append(self.WARNING_DEPENDENCY_MISSING)
        tables = self._from_native_finder(page, page_number=page_number, document_id=document_id)
        if tables:
            return tables
        tables = self._from_grid(page, page_number=page_number, document_id=document_id)
        if tables:
            return tables
        return self._from_text_rows(page, page_number=page_number, document_id=document_id)

    def from_rows(
        self,
        rows: Sequence[Sequence[Any]],
        *,
        page_number: int,
        document_id: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        extraction_method: str = "fake",
        confidence: float = 1.0,
    ) -> Table:
        values = [[self._clean_text(cell) for cell in row] for row in rows if row]
        return self._build_table(
            values,
            page_number=page_number,
            document_id=document_id,
            bbox=bbox,
            extraction_method=extraction_method,
            confidence=confidence,
        )

    def _from_native_finder(self, page: Any, *, page_number: int, document_id: str) -> List[Table]:
        finder = getattr(page, "find_tables", None)
        if not callable(finder):
            return []
        try:
            result = finder()
            native_tables = getattr(result, "tables", result or [])
            tables = []
            for index, native in enumerate(native_tables, start=1):
                extract = getattr(native, "extract", None)
                rows = extract() if callable(extract) else None
                if rows:
                    tables.append(
                        self.from_rows(
                            rows,
                            page_number=page_number,
                            document_id=document_id,
                            bbox=self._bbox(getattr(native, "bbox", None)),
                            extraction_method="pymupdf_find_tables",
                            confidence=0.95,
                        )
                    )
            return tables
        except Exception:
            self.last_warnings.append(self.WARNING_FAILED)
            return []

    def _from_grid(self, page: Any, *, page_number: int, document_id: str) -> List[Table]:
        words = getattr(page, "get_text", lambda *_: "")("words") or []
        if not words:
            return []
        horizontal, vertical = self._line_boundaries(page)
        if len(horizontal) < 2 or len(vertical) < 2:
            return []
        rows: List[List[str]] = []
        cells: List[TableCell] = []
        for row_index, (y0, y1) in enumerate(zip(horizontal, horizontal[1:])):
            row_values: List[str] = []
            for col_index, (x0, x1) in enumerate(zip(vertical, vertical[1:])):
                selected = []
                for word in words:
                    try:
                        wx0, wy0, wx1, wy1, value = float(word[0]), float(word[1]), float(word[2]), float(word[3]), str(word[4])
                    except (TypeError, ValueError, IndexError):
                        continue
                    center_x, center_y = (wx0 + wx1) / 2, (wy0 + wy1) / 2
                    if x0 - 1 <= center_x <= x1 + 1 and y0 - 1 <= center_y <= y1 + 1:
                        selected.append((wx0, value))
                text = " ".join(value for _, value in sorted(selected)).strip()
                row_values.append(text)
                cells.append(
                    TableCell(
                        row_index=row_index,
                        col_index=col_index,
                        text=text,
                        bbox=(x0, y0, x1, y1),
                        confidence=0.9,
                        is_header=row_index == 0,
                    )
                )
            if any(row_values):
                rows.append(row_values)
        if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
            return []
        table = self._build_table(
            rows,
            page_number=page_number,
            document_id=document_id,
            bbox=(vertical[0], horizontal[0], vertical[-1], horizontal[-1]),
            extraction_method="pymupdf_grid",
            confidence=0.9,
        )
        table.cells = cells
        return [table]

    def _from_text_rows(self, page: Any, *, page_number: int, document_id: str) -> List[Table]:
        raw = getattr(page, "get_text", lambda *_: "")("text") or ""
        candidates = []
        for line in raw.splitlines():
            if "|" in line:
                parts = [part.strip() for part in line.strip().strip("|").split("|")]
            elif "\t" in line:
                parts = [part.strip() for part in line.split("\t")]
            elif re.search(r"\s{2,}", line):
                parts = [part.strip() for part in re.split(r"\s{2,}", line.strip())]
            else:
                parts = []
            if len(parts) >= 2:
                candidates.append(parts)
        if len(candidates) < 2 or max(map(len, candidates)) < 2:
            return []
        return [
            self._build_table(
                candidates,
                page_number=page_number,
                document_id=document_id,
                extraction_method="text_fallback",
                confidence=0.55,
            )
        ]

    @staticmethod
    def _line_boundaries(page: Any) -> Tuple[List[float], List[float]]:
        horizontal: List[float] = []
        vertical: List[float] = []
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []
        for drawing in drawings:
            for item in drawing.get("items", []):
                if not item:
                    continue
                kind = item[0]
                if kind == "l" and len(item) >= 3:
                    p0, p1 = item[1], item[2]
                    x0, y0 = _point_xy(p0)
                    x1, y1 = _point_xy(p1)
                    if abs(y1 - y0) <= 3:
                        horizontal.extend([y0, y1])
                    if abs(x1 - x0) <= 3:
                        vertical.extend([x0, x1])
            rect = drawing.get("rect")
            if rect is not None:
                x0, y0, x1, y1 = float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
                if x1 - x0 > 5 and y1 - y0 > 5:
                    horizontal.extend([y0, y1])
                    vertical.extend([x0, x1])
        return _unique_sorted(horizontal), _unique_sorted(vertical)

    def _build_table(
        self,
        rows: Sequence[Sequence[str]],
        *,
        page_number: int,
        document_id: str,
        bbox: Optional[Tuple[float, float, float, float]],
        extraction_method: str,
        confidence: float,
    ) -> Table:
        normalized = [list(row) for row in rows]
        width = max((len(row) for row in normalized), default=0)
        normalized = [row + [""] * (width - len(row)) for row in normalized]
        headers = normalized[0] if normalized else []
        data_rows = normalized[1:] if normalized else []
        cells = [
            TableCell(
                row_index=row_index,
                col_index=col_index,
                text=value,
                bbox=(
                    bbox[0] + (bbox[2] - bbox[0]) * col_index / max(1, width),
                    bbox[1] + (bbox[3] - bbox[1]) * row_index / max(1, len(normalized)),
                    bbox[0] + (bbox[2] - bbox[0]) * (col_index + 1) / max(1, width),
                    bbox[1] + (bbox[3] - bbox[1]) * (row_index + 1) / max(1, len(normalized)),
                )
                if bbox
                else None,
                confidence=confidence,
                is_header=row_index == 0,
            )
            for row_index, row in enumerate(normalized)
            for col_index, value in enumerate(row)
        ]
        markdown = self._markdown(normalized) if self.extract_markdown else ""
        return Table(
            table_id=(
                f"{document_id}_p{page_number}_t"
                f"{hashlib.sha256(markdown.encode('utf-8')).hexdigest()[:8]}"
            ),
            page_number=page_number,
            bbox=bbox,
            headers=headers,
            rows=data_rows,
            cells=cells,
            markdown=markdown,
            plain_text="\n".join("\t".join(row) for row in normalized),
            confidence=confidence,
            extraction_method=extraction_method,
            metadata={},
        )

    @staticmethod
    def _markdown(rows: Sequence[Sequence[str]]) -> str:
        if not rows:
            return ""
        width = max(len(row) for row in rows)
        normalized = [list(row) + [""] * (width - len(row)) for row in rows]
        lines = [
            "| " + " | ".join(normalized[0]) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
        return "\n".join(lines)

    @staticmethod
    def _clean_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def _bbox(value: Any):
        try:
            return tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _document_id(path: Path) -> str:
        return f"doc_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"

    def _to_parsed_document(self, path: Path, document_id: str, tables: List[Table]):
        from src.libs.loader.parsed_document import Block, Page, ParsedDocument

        blocks = [
            Block(
                block_id=table.table_id,
                block_type="table",
                text=table.plain_text,
                page_number=table.page_number,
                bbox=table.bbox,
                metadata={"table_id": table.table_id},
            )
            for table in tables
        ]
        return ParsedDocument(
            document_id=document_id,
            source_path=str(path),
            source_type="pdf",
            pages=[Page(page_number=table.page_number, tables=[table], blocks=[block]) for table, block in zip(tables, blocks)],
            blocks=blocks,
            tables=tables,
            reading_order=[block.block_id for block in blocks],
            extraction_quality=ExtractionQuality(
                table_extraction_success=not bool(self.last_warnings),
                warnings=list(dict.fromkeys(self.last_warnings)),
            ),
            text="\n\n".join(table.plain_text for table in tables),
        )


def _unique_sorted(values: Iterable[float], tolerance: float = 2.0) -> List[float]:
    result: List[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def _point_xy(point: Any) -> Tuple[float, float]:
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])
