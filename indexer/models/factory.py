"""Embedding backend factory.

Composition roots call ``build_embedder(cfg, role)`` to obtain the configured
garment or scene embedder. Implementation modules (torch/transformers/open_clip)
are imported lazily inside the build function so importing this factory stays
cheap.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Config
    from indexer.models.base import EmbeddingModel

log = logging.getLogger(__name__)

_ROLES = ("garment", "scene")
_CLIP_BACKENDS = ("fashion-clip", "clip-vit-b32")


def build_embedder(cfg: "Config", role: str) -> "EmbeddingModel":
    """Build the embedder configured for ``role`` ("garment" or "scene").

    Reads ``embedding.{role}_backend`` and looks the backend key up in
    ``embedding.models`` for its ``model_id`` and ``dim``. The returned
    embedder's ``.name`` is the backend key.
    """
    if role not in _ROLES:
        raise ValueError(f"unknown embedder role {role!r}; expected one of {_ROLES}")

    backend = cfg.get_path(f"embedding.{role}_backend")
    if not backend:
        raise ValueError(f"config missing embedding.{role}_backend")

    spec = cfg.get_path(f"embedding.models.{backend}")
    if not spec:
        raise ValueError(f"config missing embedding.models.{backend}")
    model_id = str(spec["model_id"])
    dim = int(spec["dim"])
    device = str(cfg.get_path("embedding.device", "auto"))
    batch_size = int(cfg.get_path("embedding.batch_size", 16))

    log.info("building %s embedder: backend=%s model_id=%s dim=%d",
             role, backend, model_id, dim)

    if backend in _CLIP_BACKENDS:
        from indexer.models.clip_models import CLIPEmbedder
        return CLIPEmbedder(name=backend, model_id=model_id, dim=dim,
                            device=device, batch_size=batch_size)
    if backend == "marqo-siglip":
        from indexer.models.marqo_siglip import MarqoSiglipEmbedder
        return MarqoSiglipEmbedder(name=backend, model_id=model_id, dim=dim,
                                   device=device, batch_size=batch_size)
    raise ValueError(f"unknown embedding backend {backend!r}")
