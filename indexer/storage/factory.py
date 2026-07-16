"""Storage backend factory: config -> VectorStore.

Composition roots call :func:`build_store` with a loaded config; implementation
modules (and their heavy/optional deps — faiss, opensearchpy, pymongo) are
imported lazily inside the function so importing this module stays cheap.

Embedding dimensionalities default from the configured embedding backends
(``embedding.models.<backend>.dim``) but can be overridden explicitly, e.g.
when the caller already holds instantiated embedders.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # types only — keep imports lazy at runtime
    from core.config import Config
    from indexer.storage.base import VectorStore

log = logging.getLogger(__name__)


def _dim_from_cfg(cfg: "Config", backend_key: str) -> int:
    backend_name = cfg.get_path(f"embedding.{backend_key}")
    dim = cfg.get_path(f"embedding.models.{backend_name}.dim")
    if dim is None:
        raise ValueError(
            f"cannot resolve embedding dim: embedding.{backend_key}="
            f"{backend_name!r} has no embedding.models.{backend_name}.dim entry")
    return int(dim)


def build_store(cfg: "Config", garment_dim: int | None = None,
                scene_dim: int | None = None) -> "VectorStore":
    """Build the configured vector store (``storage.backend``; faiss default).

    Dims are resolved from ``embedding.models[embedding.garment_backend].dim``
    and ``embedding.models[embedding.scene_backend].dim`` unless given.
    """
    if garment_dim is None:
        garment_dim = _dim_from_cfg(cfg, "garment_backend")
    if scene_dim is None:
        scene_dim = _dim_from_cfg(cfg, "scene_backend")

    backend = str(cfg.get_path("storage.backend", "faiss") or "faiss")
    log.info("building vector store backend=%s garment_dim=%d scene_dim=%d",
             backend, garment_dim, scene_dim)

    if backend == "faiss":
        from indexer.storage.faiss_store import FaissStore
        return FaissStore(garment_dim=garment_dim, scene_dim=scene_dim)

    if backend == "opensearch":
        from indexer.storage.opensearch_store import OpenSearchStore
        return OpenSearchStore(
            endpoint=str(cfg.get_path("storage.opensearch.endpoint", "") or ""),
            index_prefix=str(cfg.get_path("storage.opensearch.index_prefix",
                                          "fashion") or "fashion"),
            garment_dim=garment_dim,
            scene_dim=scene_dim,
            username=os.environ.get("OPENSEARCH_USERNAME"),
            password=os.environ.get("OPENSEARCH_PASSWORD"),
        )

    if backend == "atlas":
        from indexer.storage.atlas_store import AtlasStore
        uri_env = str(cfg.get_path("storage.atlas.uri_env", "ATLAS_URI")
                      or "ATLAS_URI")
        return AtlasStore(
            uri=os.environ.get(uri_env),
            db=str(cfg.get_path("storage.atlas.db", "fashion") or "fashion"),
            garment_dim=garment_dim,
            scene_dim=scene_dim,
        )

    raise ValueError(f"unknown storage backend: {backend!r} "
                     "(expected one of: faiss, opensearch, atlas)")
