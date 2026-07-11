"""Shared terminology-aware tokenizer for query and sparse indexing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import jieba


@dataclass
class TokenizerConfig:
    user_dict_path: str = ""
    stopwords_path: str = ""
    preserve_patterns: list[str] = field(default_factory=lambda: [r"[A-Za-z]+[A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+", r"[A-Za-z]+\d+(?:\.\d+)?"])
    enable_char_ngram: bool = False
    char_ngram_min: int = 2
    char_ngram_max: int = 3


class DomainTokenizer:
    def __init__(self, config: TokenizerConfig | None = None, stopwords: Iterable[str] = ()) -> None:
        self.config = config or TokenizerConfig()
        self.stopwords = {str(item).strip().lower() for item in stopwords if str(item).strip()}
        if self.config.stopwords_path and Path(self.config.stopwords_path).exists():
            self.stopwords.update(Path(self.config.stopwords_path).read_text(encoding="utf-8").splitlines())
        if self.config.user_dict_path and Path(self.config.user_dict_path).exists():
            jieba.load_userdict(self.config.user_dict_path)
        self.protected = [re.compile(pattern) for pattern in self.config.preserve_patterns]

    def tokenize(self, text: str) -> list[str]:
        value = str(text or "")
        protected_matches: list[tuple[int, int, str]] = []
        for pattern in self.protected:
            protected_matches.extend(
                (match.start(), match.end(), match.group(0))
                for match in pattern.finditer(value)
            )

        # Keep protected terminology as an atomic token.  Replacing it with a
        # textual placeholder is not reliable because jieba splits underscores
        # and digits in placeholders (for example ``__TERM_0_0__``).
        protected_matches.sort(key=lambda item: (item[0], -item[1]))
        non_overlapping: list[tuple[int, int, str]] = []
        for match in protected_matches:
            if non_overlapping and match[0] < non_overlapping[-1][1]:
                continue
            non_overlapping.append(match)

        tokens: list[str] = []
        cursor = 0
        for start, end, term in non_overlapping:
            tokens.extend(self._tokenize_plain(value[cursor:start]))
            tokens.append(term)
            cursor = end
        tokens.extend(self._tokenize_plain(value[cursor:]))

        if self.config.enable_char_ngram:
            for token in list(tokens):
                if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                    for size in range(self.config.char_ngram_min, self.config.char_ngram_max + 1):
                        tokens.extend(token[index : index + size] for index in range(len(token) - size + 1))
        return tokens

    def _tokenize_plain(self, value: str) -> list[str]:
        tokens: list[str] = []
        for item in jieba.lcut(value):
            item = item.strip()
            if not item or item.lower() in self.stopwords or re.fullmatch(r"[\W_]+", item, re.UNICODE):
                continue
            tokens.append(item)
        return tokens
