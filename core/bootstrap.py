"""Composition root: wire the store, indexing pipeline and search engine from config.

Scripts, the eval harness and the API layer construct the system exclusively
through these functions rather than touching backend factories directly. Heavy
factories and implementation modules are imported lazily inside each function
so importing ``core.bootstrap`` stays cheap.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

from core.config import Config, resolve

if TYPE_CHECKING:
    from PIL.Image import Image

    from indexer.pipeline import IndexingPipeline
    from indexer.storage.base import VectorStore
    from retriever.search.engine import SearchEngine

log = logging.getLogger(__name__)


def load_store(cfg: Config) -> "VectorStore":
    """Build the configured vector store and load persisted artifacts if present."""
    from indexer.storage.factory import build_store

    store = build_store(cfg)
    index_dir = resolve(cfg, "paths.artifacts_dir") / "index"
    if index_dir.exists():
        store.load(index_dir)
        log.info("loaded index from %s (%d images)", index_dir, len(store.image_ids()))
    else:
        log.info("no persisted index at %s — store starts empty", index_dir)
    return store


def default_image_resolver(cfg: Config) -> Callable[[str], "Image"]:
    """Return a resolver: image_id -> PIL RGB image from cfg paths.images_dir."""
    images_dir = resolve(cfg, "paths.images_dir")

    def _resolve(image_id: str) -> "Image":
        from PIL import Image as PILImage

        with PILImage.open(images_dir / image_id) as im:
            return im.convert("RGB")

    return _resolve


def build_engine(
    cfg: Config,
    store: "Optional[VectorStore]" = None,
    with_reranker: bool = True,
    image_resolver: Optional[Callable[[str], "Image"]] = None,
) -> "SearchEngine":
    """Wire the full retrieval stack: parser, embedders, store, optional reranker.

    ``store=None`` loads the persisted store from artifacts; pass an existing
    store to share one across engine and pipeline. ``with_reranker=False``
    skips the reranker even when enabled in config (e.g. lightweight eval runs).
    """
    from indexer.models.factory import build_embedder
    from retriever.parsing.factory import build_parser
    from retriever.search.engine import SearchEngine

    if store is None:
        store = load_store(cfg)

    reranker = None
    if with_reranker and cfg.get_path("rerank.enabled", False):
        from retriever.rerank.factory import build_reranker

        reranker = build_reranker(cfg)

    weights = cfg.get_path("search.weights")
    return SearchEngine(
        store=store,
        garment_embedder=build_embedder(cfg, "garment"),
        scene_embedder=build_embedder(cfg, "scene"),
        parser=build_parser(cfg),
        reranker=reranker,
        image_resolver=image_resolver or default_image_resolver(cfg),
        weights=dict(weights) if weights else None,
        candidates_k=int(cfg.get_path("search.candidates_k", 50)),
        prefilter_enabled=bool(cfg.get_path("search.prefilter.enabled", True)),
        prefilter_min_candidates=int(cfg.get_path("search.prefilter.min_candidates", 12)),
        prefilter_soft_fallback=bool(cfg.get_path("search.prefilter.soft_fallback", True)),
        rerank_top_n=int(cfg.get_path("rerank.top_n", 50)),
        rerank_weight=float(cfg.get_path("rerank.weight", 0.35)),
    )


def build_pipeline(cfg: Config) -> "IndexingPipeline":
    """Wire the indexing pipeline: detector, embedders, extractor, store."""
    from indexer.detection.factory import build_detector
    from indexer.extraction.factory import build_extractor
    from indexer.models.factory import build_embedder
    from indexer.pipeline import IndexingPipeline

    return IndexingPipeline(
        detector=build_detector(cfg),
        garment_embedder=build_embedder(cfg, "garment"),
        scene_embedder=build_embedder(cfg, "scene"),
        extractor=build_extractor(cfg),
        store=load_store(cfg),
        artifacts_dir=resolve(cfg, "paths.artifacts_dir"),
    )
