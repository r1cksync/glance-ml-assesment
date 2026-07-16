"""Query parser interface + caching wrapper."""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Optional, Protocol

from core.schemas import ParsedQuery


class QueryParser(ABC):
    name: str

    @abstractmethod
    def parse(self, query: str) -> ParsedQuery:
        """Decompose a natural-language query into structured intent."""


class ParseCache(Protocol):
    """Minimal cache protocol — memory impl here, DynamoDB impl in the API layer."""
    def get(self, key: str) -> Optional[str]: ...
    def put(self, key: str, value: str) -> None: ...


class MemoryParseCache:
    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._d.get(key)

    def put(self, key: str, value: str) -> None:
        self._d[key] = value


class CachedParser(QueryParser):
    """Wraps any parser with a string cache keyed on (parser, normalized query)."""

    def __init__(self, inner: QueryParser, cache: ParseCache):
        self.inner = inner
        self.cache = cache
        self.name = f"cached({inner.name})"

    @staticmethod
    def _key(parser_name: str, query: str) -> str:
        norm = " ".join(query.strip().lower().split())
        return hashlib.sha256(f"{parser_name}::{norm}".encode()).hexdigest()

    def parse(self, query: str) -> ParsedQuery:
        key = self._key(self.inner.name, query)
        hit = self.cache.get(key)
        if hit:
            return ParsedQuery.model_validate(json.loads(hit))
        parsed = self.inner.parse(query)
        self.cache.put(key, json.dumps(parsed.model_dump()))
        return parsed
