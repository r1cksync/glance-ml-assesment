"""Factory for attribute extractors.

Importing this module is cheap: implementation modules (which pull in torch,
transformers, or boto3) are imported lazily inside `build_extractor`, keyed by
`cfg.extraction.backend`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Config
    from indexer.extraction.base import AttributeExtractor

log = logging.getLogger(__name__)


def build_extractor(cfg: "Config") -> "AttributeExtractor":
    """Build the configured attribute extractor.

    Dispatches on cfg.extraction.backend in {qwen2vl, moondream, bedrock},
    passing cfg.extraction.models[backend] params, cfg.extraction.max_retries,
    and (for bedrock) cfg.app.region.
    """
    backend = str(cfg.get_path("extraction.backend", "qwen2vl"))
    params = dict(cfg.get_path(f"extraction.models.{backend}") or {})
    max_retries = int(cfg.get_path("extraction.max_retries", 2))
    device = str(cfg.get_path("extraction.device", "auto"))
    log.info("building extraction backend %r", backend)

    if backend == "qwen2vl":
        from indexer.extraction.qwen2vl import Qwen2VLExtractor
        return Qwen2VLExtractor(
            model_id=str(params["model_id"]),
            quant=str(params.get("quant", "4bit")),
            max_new_tokens=int(params.get("max_new_tokens", 512)),
            device=device,
            max_retries=max_retries,
        )
    if backend == "moondream":
        from indexer.extraction.moondream import MoondreamExtractor
        return MoondreamExtractor(
            model_id=str(params["model_id"]),
            revision=str(params["revision"]),
            device=device,
            max_retries=max_retries,
        )
    if backend == "bedrock":
        from indexer.extraction.bedrock import BedrockExtractor
        return BedrockExtractor(
            model_id=str(params["model_id"]),
            region=str(cfg.get_path("app.region", "us-east-1")),
            max_retries=max_retries,
        )
    raise ValueError(f"unknown extraction backend: {backend!r}")
