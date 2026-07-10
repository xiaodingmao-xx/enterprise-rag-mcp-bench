"""Repeated page header/footer detection for structured PDF blocks."""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from src.libs.loader.parsed_document import Block, Page


class HeaderFooterDetector:
    def __init__(
        self,
        *,
        similarity_threshold: float = 0.85,
        min_repeat_pages: int = 2,
        top_margin_ratio: float = 0.12,
        bottom_margin_ratio: float = 0.12,
    ) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.min_repeat_pages = max(2, int(min_repeat_pages))
        self.top_margin_ratio = float(top_margin_ratio)
        self.bottom_margin_ratio = float(bottom_margin_ratio)

    def detect(self, pages: List[Page]) -> Tuple[List[Block], List[Block], List[Block], List[Block]]:
        if len(pages) < 2:
            body = [block for page in pages for block in page.blocks]
            return [], [], body, []

        top_candidates: Dict[str, List[Block]] = defaultdict(list)
        bottom_candidates: Dict[str, List[Block]] = defaultdict(list)
        for page in pages:
            for block in page.blocks:
                if not block.text or not block.bbox or not page.height:
                    continue
                top_ratio = block.bbox[1] / page.height
                bottom_ratio = (page.height - block.bbox[3]) / page.height
                normalized = self._normalise(block.text)
                if not normalized:
                    continue
                if top_ratio <= self.top_margin_ratio:
                    top_candidates[normalized].append(block)
                if bottom_ratio <= self.bottom_margin_ratio:
                    bottom_candidates[normalized].append(block)

        header_keys = self._repeated_keys(top_candidates)
        footer_keys = self._repeated_keys(bottom_candidates)
        # Page number footers naturally differ by page; identify them using a
        # stable pattern while still requiring at least two pages.
        page_number_footers = [
            block
            for page in pages
            for block in page.blocks
            if block.text
            and block.bbox
            and page.height
            and (page.height - block.bbox[3]) / page.height <= self.bottom_margin_ratio
            and self._is_page_number_footer(block.text)
        ]

        headers = self._mark(header_keys, top_candidates, "header")
        footers = self._mark(footer_keys, bottom_candidates, "footer")
        for block in page_number_footers:
            if block not in footers:
                block.block_type = "footer"
                block.metadata["header_footer_reason"] = "page_number"
                footers.append(block)

        excluded_ids = {id(block) for block in headers + footers}
        body = [
            block
            for page in pages
            for block in page.blocks
            if id(block) not in excluded_ids
        ]
        warnings = ["HEADER_FOOTER_DETECTED"] if headers or footers else []
        return headers, footers, body, warnings

    def _repeated_keys(self, candidates: Dict[str, List[Block]]) -> set[str]:
        keys = list(candidates)
        repeated = set()
        for key in keys:
            pages = {
                block.page_number
                for other_key in keys
                if self._similar(key, other_key)
                for block in candidates[other_key]
            }
            if len(pages) >= self.min_repeat_pages:
                repeated.add(key)
        return repeated

    def _mark(self, keys: set[str], candidates: Dict[str, List[Block]], block_type: str) -> List[Block]:
        result: List[Block] = []
        for candidate_key, candidate_blocks in candidates.items():
            if not any(self._similar(candidate_key, key) for key in keys):
                continue
            for block in candidate_blocks:
                block.block_type = block_type
                block.metadata["header_footer_reason"] = "repeated_text"
                result.append(block)
        return result

    def _similar(self, left: str, right: str) -> bool:
        return SequenceMatcher(None, left, right).ratio() >= self.similarity_threshold

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _is_page_number_footer(text: str) -> bool:
        value = re.sub(r"\s+", " ", text).strip()
        return bool(
            re.fullmatch(r"(?:page\s*)?\d+(?:\s*/\s*\d+)?", value, re.I)
            or re.fullmatch(r"第\s*\d+\s*(?:页|/\s*\d+)?", value)
        )
