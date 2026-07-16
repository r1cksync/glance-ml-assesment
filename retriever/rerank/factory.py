"""Reranker factory: config name -> constructed Reranker (or None when disabled).

Implementation modules (torch/transformers) are imported lazily inside the
build function so importing this factory stays cheap.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.config import Config
    from retriever.rerank.base import Reranker

log = logging.getLogger(__name__)


def build_reranker(cfg: "Config") -> "Optional[Reranker]":
    """Build the configured reranker. Returns None when cfg.rerank.enabled is false."""
    if not cfg.get_path("rerank.enabled", False):
        log.info("reranking disabled in config")
        return None
    backend = cfg.get_path("rerank.backend", "blip-itm")
    if backend == "blip-itm":
        from retriever.rerank.blip_itm import BlipITMReranker
        return BlipITMReranker(
            model_id=str(cfg.get_path("rerank.model_id", "Salesforce/blip-itm-base-coco")),
            device=str(cfg.get_path("rerank.device", "auto")),
            batch_size=int(cfg.get_path("rerank.batch_size", 8)),
        )
    raise ValueError(f"unknown rerank backend: {backend!r}")
