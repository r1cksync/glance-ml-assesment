"""Marqo FashionSigLIP dual encoder loaded through open_clip's hf-hub bridge.

``Marqo/marqo-fashionSigLIP`` (dim 768) ships an open_clip-compatible config,
so ``open_clip.create_model_and_transforms("hf-hub:...")`` returns the model,
its image preprocess transform, and a matching tokenizer. Outputs follow the
EmbeddingModel contract: L2-normalized float32 numpy arrays.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
import open_clip
import torch

from indexer.models.base import EmbeddingModel
from indexer.models.device import resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)


class MarqoSiglipEmbedder(EmbeddingModel):
    """Batched image/text embedder over an open_clip hf-hub SigLIP checkpoint."""

    def __init__(self, name: str, model_id: str, dim: int,
                 device: str = "auto", batch_size: int = 16) -> None:
        self.name = name
        self.model_id = model_id
        self.dim = dim
        self.batch_size = max(1, int(batch_size))
        self.device = resolve_device(device)

        hub_ref = f"hf-hub:{model_id}"
        log.info("loading open_clip checkpoint %s (dim=%d) on %s",
                 hub_ref, dim, self.device)
        model, _, preprocess = open_clip.create_model_and_transforms(hub_ref)
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(hub_ref)

    # ── EmbeddingModel API ────────────────────────────────────────────────────

    def embed_images(self, images: Sequence["Image"]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start:start + self.batch_size]
                pixel_values = torch.stack([
                    self.preprocess(im if im.mode == "RGB" else im.convert("RGB"))
                    for im in batch
                ]).to(self.device)
                feats = self.model.encode_image(pixel_values)
                chunks.append(feats.float().cpu().numpy())
        return self._normalize(np.concatenate(chunks, axis=0))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start:start + self.batch_size])
                tokens = self.tokenizer(batch).to(self.device)
                feats = self.model.encode_text(tokens)
                chunks.append(feats.float().cpu().numpy())
        return self._normalize(np.concatenate(chunks, axis=0))
