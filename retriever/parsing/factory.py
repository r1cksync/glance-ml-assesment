"""Query-parser factory: config name -> constructed QueryParser (optionally cached).

Parsers are wrapped in CachedParser with the in-process MemoryParseCache when
cfg.parsing.cache.enabled — that is the right cache for scripts and eval runs.
The API layer injects its own DynamoDB-backed ParseCache implementation instead
(same ParseCache protocol), so no AWS coupling exists here.

Implementation modules are imported lazily inside the build function so
importing this factory stays cheap.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Config
    from retriever.parsing.base import QueryParser

log = logging.getLogger(__name__)


def build_parser(cfg: "Config", cache: "object | None" = None) -> "QueryParser":
    """Build the configured query parser (rule | bedrock), cached if enabled.

    ``cache`` overrides the default in-process cache with any ParseCache
    implementation (the API layer passes its DynamoDB-backed cache here).
    """
    backend = cfg.get_path("parsing.backend", "rule")
    if backend == "rule":
        from retriever.parsing.rule_parser import RuleParser
        parser: "QueryParser" = RuleParser()
    elif backend == "bedrock":
        from retriever.parsing.bedrock_parser import BedrockParser
        parser = BedrockParser(
            model_id=str(cfg.get_path("parsing.bedrock_model_id")),
            region=str(cfg.get_path("app.region", "us-east-1")),
        )
    else:
        raise ValueError(f"unknown parsing backend: {backend!r}")

    if cache is not None:
        from retriever.parsing.base import CachedParser
        parser = CachedParser(parser, cache)  # type: ignore[arg-type]
    elif cfg.get_path("parsing.cache.enabled", False):
        from retriever.parsing.base import CachedParser, MemoryParseCache
        parser = CachedParser(parser, MemoryParseCache())
    log.info("query parser: %s", parser.name)
    return parser
